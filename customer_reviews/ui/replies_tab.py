"""Replies tab: review, edit and approve AI-suggested replies."""

import pandas as pd
import streamlit as st

from customer_reviews.core.safety import make_csv_safe
from customer_reviews.storage import REPLY_STATUSES, Database


EDITOR_KEY = "reply_editor"

EDITABLE = ["reply_status", "reply_text"]
SHOWN = ["review_id", "platform", "rating", "urgency", "language", "summary", "review_text"]


def _needs_reply(df: pd.DataFrame) -> pd.Series:
    return (df["sentiment"] == "Negative") | df["urgency"].isin(["HIGH", "MEDIUM"])


def build_reply_table(df: pd.DataFrame, replies: dict[str, dict]) -> pd.DataFrame:
    """Reviews with their current reply status and (edited or suggested) reply."""

    table = df[SHOWN].copy()

    saved = [replies.get(review_id, {}) for review_id in df["review_id"]]

    table["reply_status"] = [entry.get("reply_status") or "Open" for entry in saved]
    table["reply_text"] = [
        entry.get("reply_text") or suggested
        for entry, suggested in zip(saved, df["suggested_reply"], strict=True)
    ]

    return table.reset_index(drop=True)


def find_changes(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, dict]:
    """Rows whose status or reply text was edited."""

    changes = {}

    for old, new in zip(before.to_dict("records"), after.to_dict("records"), strict=True):
        if any(old[column] != new[column] for column in EDITABLE):
            changes[new["review_id"]] = {column: new[column] for column in EDITABLE}

    return changes


def render_replies_tab(df: pd.DataFrame, db: Database | None) -> None:
    replies: dict = st.session_state.replies

    show_all = st.toggle(
        "Show all reviews",
        help="By default, only negative reviews and high/medium urgency reviews are listed.",
    )

    subset = df if show_all else df[_needs_reply(df)]

    if subset.empty:
        st.success("No reviews need a reply.")
        return

    table = build_reply_table(subset, replies)

    counts = table["reply_status"].value_counts()
    cols = st.columns(len(REPLY_STATUSES))
    for col, status in zip(cols, REPLY_STATUSES, strict=True):
        col.metric(status, int(counts.get(status, 0)))

    st.caption(
        "Edit the reply text and set the status. Replies are AI-generated: check "
        "them before sending."
        + (" Changes are saved automatically." if db is not None else "")
    )

    editor_key = f"{EDITOR_KEY}_{'all' if show_all else 'needs_reply'}"

    edited = st.data_editor(
        table,
        key=editor_key,
        hide_index=True,
        disabled=[c for c in table.columns if c not in EDITABLE],
        column_order=["reply_status", "review_id", "platform", "rating", "urgency",
                      "language", "summary", "reply_text", "review_text"],
        column_config={
            "reply_status": st.column_config.SelectboxColumn(
                "Status", options=REPLY_STATUSES, required=True
            ),
            "reply_text": st.column_config.TextColumn("Reply", width="large"),
            "summary": st.column_config.TextColumn("Summary", width="medium"),
            "review_text": st.column_config.TextColumn("Review", width="medium"),
        },
    )

    changes = find_changes(table, edited)

    if changes:
        replies.update(changes)

        if db is not None:
            db.update_replies(changes)

        # Reset the editor so its pending edits are not re-applied to the
        # updated table (or to the other view) on later runs.
        del st.session_state[editor_key]
        st.rerun()

    approved = edited[edited["reply_status"] == "Approved"]

    st.download_button(
        label=f"Download approved replies ({len(approved)})",
        data=make_csv_safe(approved[["review_id", "platform", "reply_text"]]).to_csv(index=False),
        file_name="approved_replies.csv",
        mime="text/csv",
        disabled=approved.empty,
        icon=":material/download:",
    )

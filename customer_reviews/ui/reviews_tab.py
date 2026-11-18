"""Urgent and Explore tabs."""

import pandas as pd
import streamlit as st

from customer_reviews.core.safety import make_csv_safe


URGENT_COLUMNS = [
    "review_id", "date", "platform", "rating", "category", "summary", "suggested_reply",
]

EXPLORE_COLUMNS = [
    "review_id", "date", "platform", "rating", "sentiment", "category", "urgency",
    "summary", "suggested_reply", "review_text",
]

COLUMN_CONFIG = {
    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
    "summary": st.column_config.TextColumn("Summary", width="large"),
    "suggested_reply": st.column_config.TextColumn("Suggested reply", width="large"),
    "review_text": st.column_config.TextColumn("Review", width="large"),
}


def render_urgent_tab(df: pd.DataFrame) -> None:
    """All high-urgency complaints (sidebar filters do not apply)."""

    urgent = df[df["urgency"] == "HIGH"]

    if urgent.empty:
        st.success("No high urgency complaints found.")
        return

    st.error(f"{len(urgent)} high urgency complaint(s) require human review.", icon="🚨")
    st.caption(
        "Shows every high-urgency review, regardless of sidebar filters. Suggested "
        "replies are AI-generated: review and edit them before sending."
    )

    st.dataframe(urgent, hide_index=True, column_order=URGENT_COLUMNS, column_config=COLUMN_CONFIG)

    st.download_button(
        label="Download urgent complaints (CSV)",
        data=make_csv_safe(urgent[URGENT_COLUMNS]).to_csv(index=False),
        file_name="urgent_complaints.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_explore_tab(filtered: pd.DataFrame, total_reviews: int) -> None:
    st.write(f"**{len(filtered)}** of {total_reviews} analysed reviews match the filters.")

    st.dataframe(
        filtered, hide_index=True, column_order=EXPLORE_COLUMNS, column_config=COLUMN_CONFIG
    )

    st.download_button(
        label="Download filtered reviews (CSV)",
        data=make_csv_safe(filtered).to_csv(index=False),
        file_name="filtered_reviews.csv",
        mime="text/csv",
        disabled=filtered.empty,
        icon=":material/download:",
    )

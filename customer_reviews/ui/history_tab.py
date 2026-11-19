"""History tab: all reviews saved in the local database."""

import streamlit as st

from customer_reviews.core.safety import make_csv_safe
from customer_reviews.storage import Database
from customer_reviews.ui import charts
from customer_reviews.ui.reviews_tab import COLUMN_CONFIG


HISTORY_COLUMNS = [
    "review_id", "date", "platform", "rating", "language", "sentiment", "category",
    "aspects", "urgency", "summary", "reply_status", "reply_text", "review_text",
    "model_id", "analysed_at",
]


def render_history_tab(db: Database, theme: str) -> None:
    stored = db.load_reviews()

    if stored.empty:
        st.info("No saved reviews yet. Analysed reviews are saved here automatically.")
        return

    dates = stored["date"].dropna()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Saved reviews", len(stored))
    col2.metric(
        "Date range",
        f"{dates.min():%d %b %Y} – {dates.max():%d %b %Y}" if not dates.empty else "–",
    )
    col3.metric("Negative", int((stored["sentiment"] == "Negative").sum()))
    handled = stored["reply_status"].isin(["Approved", "Sent"])
    col4.metric("Replies approved or sent", int(handled.sum()))

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(charts.weekly_sentiment_bar(stored, theme), key="history_sentiment")
    with col6:
        st.plotly_chart(charts.weekly_rating_line(stored, theme), key="history_rating")

    st.dataframe(
        stored,
        hide_index=True,
        column_order=HISTORY_COLUMNS,
        column_config={
            **COLUMN_CONFIG,
            "reply_text": st.column_config.TextColumn("Reply", width="large"),
        },
    )

    st.download_button(
        label="Download history (CSV)",
        data=make_csv_safe(stored).to_csv(index=False),
        file_name="review_history.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    with st.expander("Delete saved data"):
        st.write(
            "Permanently deletes all saved reviews, replies and cached analyses "
            f"from `{db.path.name}`. This cannot be undone."
        )
        confirmed = st.checkbox("I understand, delete everything")

        if st.button("Delete all saved data", type="primary", disabled=not confirmed):
            db.delete_all()
            st.session_state.replies = {}
            st.success("All saved data was deleted.")
            st.rerun()

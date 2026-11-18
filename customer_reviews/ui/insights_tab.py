"""Insights tab: AI-generated business insights report."""

import streamlit as st

from customer_reviews.core.safety import sanitize_markdown
from customer_reviews.services.insights import generate_weekly_insights
from customer_reviews.ui.common import friendly_error


def render_insights_tab(analysed, can_call_llm: bool) -> None:
    st.caption(
        "The report covers all successfully analysed reviews (sidebar filters "
        "do not apply)."
    )

    if st.button(
        "Generate insights report",
        disabled=not can_call_llm,
        icon=":material/insights:",
    ):
        try:
            with st.spinner("AI is preparing your business insights..."):
                report = generate_weekly_insights(analysed)

            # Strip links/images: the report is built from untrusted content
            st.session_state.weekly_report = sanitize_markdown(report)

        except Exception as e:
            st.error(friendly_error("Insights generation", e))

    report = st.session_state.weekly_report

    if report is None:
        st.info("Generate the report to see a summary, top issues, trends and recommended actions.")
        return

    st.caption("AI-generated report. Please verify key figures before sharing it.")

    with st.container(border=True):
        st.markdown(report)

    st.download_button(
        label="Download report (Markdown)",
        data=report,
        file_name="business_insights_report.md",
        mime="text/markdown",
        icon=":material/download:",
    )

"""Analysis tab: usage estimate, running the AI analysis, retries and results."""

import streamlit as st

from customer_reviews.config import Settings
from customer_reviews.core.safety import make_csv_safe
from customer_reviews.services.analysis import estimate_usage, merge_results, process_reviews
from customer_reviews.storage import Database
from customer_reviews.ui.common import (
    analysis_cache,
    friendly_error,
    remember_results,
    set_flash,
    show_flash,
)


RESULT_COLUMNS = [
    "review_id", "date", "platform", "rating", "language", "sentiment", "category",
    "aspects", "urgency", "summary", "suggested_reply", "review_text", "analysis_status",
]


def _run(settings: Settings, db: Database | None, reviews: list[dict], label: str) -> list[dict]:
    """Run the analysis with a progress bar."""

    progress = st.progress(0.0, text=label)

    def update(done: int, total: int) -> None:
        progress.progress(done / total, text=f"{label} {done} of {total} reviews done")

    results = process_reviews(
        reviews,
        cache=analysis_cache(db),
        model_id=settings.model_id,
        max_chars=settings.max_review_chars,
        max_concurrency=settings.max_concurrency,
        on_progress=update,
    )

    progress.empty()

    return results


def render_analysis_tab(settings: Settings, db: Database | None) -> None:
    valid_df = st.session_state.validated_reviews

    if valid_df is None or valid_df.empty:
        st.info("Upload a valid review CSV to start the analysis.")
        return

    show_flash()

    to_analyse = valid_df

    if len(valid_df) > settings.max_reviews:
        st.warning(
            f"The file has {len(valid_df)} valid reviews. Only the first "
            f"{settings.max_reviews} will be analysed (raise MAX_REVIEWS in .env "
            "to change the limit)."
        )
        to_analyse = valid_df.head(settings.max_reviews)

    reviews = to_analyse.to_dict(orient="records")

    estimate = estimate_usage(
        reviews,
        cache=analysis_cache(db),
        model_id=settings.model_id,
        max_chars=settings.max_review_chars,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Reviews to analyse", estimate["new_reviews"])
    col2.metric("Already analysed (cached)", estimate["cached_reviews"])
    col3.metric(
        "Estimated tokens",
        f"~{estimate['input_tokens'] + estimate['output_tokens']:,}",
        help=(
            f"About {estimate['input_tokens']:,} input and "
            f"{estimate['output_tokens']:,} output tokens for {settings.model_id}. "
            "A rough estimate: models that think before answering use more."
        ),
    )

    if st.button(
        "Analyse customer reviews",
        type="primary",
        disabled=not settings.has_api_key(),
        icon=":material/auto_awesome:",
    ):
        try:
            results = _run(settings, db, reviews, "Analysing…")
        except Exception as e:
            st.error(friendly_error("Review analysis", e))
        else:
            analysed = merge_results(to_analyse, results)
            st.session_state.analysed_reviews = analysed
            remember_results(analysed, settings, db)
            st.session_state.weekly_report = None

            succeeded = int((analysed["analysis_status"] == "success").sum())
            set_flash("success", f"Successfully analysed {succeeded} of {len(analysed)} reviews!")
            st.rerun()

    analysed = st.session_state.analysed_reviews

    if analysed is None:
        return

    failed_df = analysed[analysed["analysis_status"] == "failed"]

    if not failed_df.empty:
        st.warning(
            f"{len(failed_df)} review(s) could not be analysed. They are "
            "excluded from the dashboard and insights."
        )

        with st.expander("Show failed reviews"):
            st.dataframe(
                failed_df[["review_id", "review_text", "analysis_error"]],
                hide_index=True,
            )

        if st.button("Retry failed reviews", icon=":material/refresh:"):
            try:
                retry_results = _run(
                    settings, db, failed_df.to_dict(orient="records"), "Retrying…"
                )
            except Exception as e:
                st.error(friendly_error("Retry", e))
            else:
                analysed = merge_results(analysed, retry_results)
                st.session_state.analysed_reviews = analysed
                remember_results(analysed, settings, db)
                st.session_state.weekly_report = None
                st.rerun()

    st.subheader("Analysis results")

    st.dataframe(
        analysed,
        hide_index=True,
        column_order=[c for c in RESULT_COLUMNS if c in analysed.columns],
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "summary": st.column_config.TextColumn("Summary", width="large"),
            "suggested_reply": st.column_config.TextColumn("Suggested reply", width="large"),
            "review_text": st.column_config.TextColumn("Review", width="large"),
        },
    )

    if db is not None:
        st.caption("Results are saved to the local history database (History tab).")

    st.download_button(
        label="Download all results (CSV)",
        data=make_csv_safe(analysed).to_csv(index=False),
        file_name="customer_review_analysis.csv",
        mime="text/csv",
        icon=":material/download:",
    )

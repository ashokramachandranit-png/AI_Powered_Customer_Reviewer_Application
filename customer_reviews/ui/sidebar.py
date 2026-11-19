"""Sidebar: file upload, configuration summary and review filters."""

from pathlib import Path

import pandas as pd
import streamlit as st

from customer_reviews.config import Settings
from customer_reviews.core.validation import REQUIRED_COLUMNS
from customer_reviews.services.filters import ReviewFilters
from customer_reviews.ui.charts import SENTIMENT_ORDER, URGENCY_ORDER
from customer_reviews.ui.common import FILTER_KEY_PREFIX


SAMPLE_CSV_PATH = Path(__file__).resolve().parents[2] / "sample_data" / "sample_reviews.csv"


def render_upload(settings: Settings):
    """Render the upload controls and return the uploaded file (or None)."""

    with st.sidebar:
        st.header("Upload reviews")

        uploaded_file = st.file_uploader(
            "Customer review CSV",
            type=["csv"],
            help="Required columns: " + ", ".join(REQUIRED_COLUMNS),
        )

        if SAMPLE_CSV_PATH.exists():
            st.download_button(
                label="Download sample CSV",
                data=SAMPLE_CSV_PATH.read_bytes(),
                file_name="sample_reviews.csv",
                mime="text/csv",
                icon=":material/download:",
            )

        st.caption(
            f"Model: `{settings.model_id}` · up to {settings.max_reviews} "
            f"reviews per run. Review text is sent to {settings.llm_provider} "
            "for analysis."
        )

        if settings.database_file is not None:
            st.caption(
                "Analysed reviews and replies are saved locally in "
                f"`{settings.database_path}` (History tab)."
            )

        if not settings.has_api_key():
            st.warning(
                f"{settings.api_key_env_var} is not set. You can upload and "
                "validate a CSV, but AI analysis needs an API key in your "
                ".env file."
            )

    return uploaded_file


def _options(df: pd.DataFrame, column: str, order: list[str] | None = None) -> list[str]:
    present = set(df[column].dropna().astype(str))

    if order:
        return [value for value in order if value in present]

    return sorted(present)


def render_filters(df: pd.DataFrame) -> ReviewFilters:
    """Render filter widgets for analysed reviews and return the selection."""

    k = FILTER_KEY_PREFIX

    with st.sidebar:
        st.divider()
        st.header("Filter reviews")
        st.caption("Filters apply to the Dashboard and Explore tabs. Leave empty for all.")

        filters = ReviewFilters(
            search=st.text_input(
                "Search text", key=k + "search", placeholder="e.g. refund, cold, staff"
            ),
            platforms=st.multiselect(
                "Platform", _options(df, "platform"), key=k + "platform", placeholder="All"
            ),
            sentiments=st.multiselect(
                "Sentiment", _options(df, "sentiment", SENTIMENT_ORDER),
                key=k + "sentiment", placeholder="All",
            ),
            categories=st.multiselect(
                "Category", _options(df, "category"), key=k + "category", placeholder="All"
            ),
            urgencies=st.multiselect(
                "Urgency", _options(df, "urgency", URGENCY_ORDER),
                key=k + "urgency", placeholder="All",
            ),
        )

        dates = pd.to_datetime(df["date"], errors="coerce").dropna()

        if not dates.empty:
            first, last = dates.min().date(), dates.max().date()

            selected = st.date_input(
                "Date range",
                value=(first, last),
                min_value=first,
                max_value=last,
                key=k + "dates",
            )

            # While picking, the widget briefly returns a single date
            if isinstance(selected, tuple) and len(selected) == 2:
                start, end = selected
                filters.date_from = start if start > first else None
                filters.date_to = end if end < last else None

    return filters

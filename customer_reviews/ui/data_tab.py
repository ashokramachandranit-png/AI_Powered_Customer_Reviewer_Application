"""Data tab: upload handling, validation report and preview."""

import hashlib

import pandas as pd
import streamlit as st

from customer_reviews.core.safety import make_csv_safe
from customer_reviews.core.validation import validate_reviews_with_report
from customer_reviews.ui.common import reset_for_new_file


def handle_upload(uploaded_file) -> None:
    """Parse and validate a newly uploaded file (only when its content changes)."""

    if uploaded_file is None:
        # File removed: forget results that belonged to it
        if st.session_state.upload_signature is not None:
            st.session_state.upload_signature = None
            reset_for_new_file()
        return

    signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

    if signature == st.session_state.upload_signature:
        return

    st.session_state.upload_signature = signature
    reset_for_new_file()

    try:
        uploaded_df = pd.read_csv(uploaded_file, dtype={"review_id": "string"})
        valid_df, rejected_df = validate_reviews_with_report(uploaded_df)

        st.session_state.validated_reviews = valid_df
        st.session_state.rejected_reviews = rejected_df

    except Exception as e:
        st.session_state.upload_error = str(e)


def render_data_tab() -> None:
    if st.session_state.upload_error:
        st.error(f"CSV validation failed: {st.session_state.upload_error}")
        return

    valid_df = st.session_state.validated_reviews
    rejected_df = st.session_state.rejected_reviews

    if valid_df is None:
        return

    if valid_df.empty:
        st.error("No valid customer reviews found in the uploaded CSV.")
    else:
        st.success(f"CSV uploaded successfully! {len(valid_df)} valid reviews found.")

    if rejected_df is not None and not rejected_df.empty:
        st.warning(f"{len(rejected_df)} row(s) were skipped because of data problems.")

        with st.expander("Show skipped rows"):
            st.dataframe(rejected_df, hide_index=True)
            st.download_button(
                label="Download skipped rows (CSV)",
                data=make_csv_safe(rejected_df).to_csv(index=False),
                file_name="skipped_reviews.csv",
                mime="text/csv",
            )

    if not valid_df.empty:
        st.subheader("Uploaded reviews")
        st.dataframe(
            valid_df,
            hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "rating": st.column_config.NumberColumn("Rating", format="%.1f ⭐"),
                "review_text": st.column_config.TextColumn("Review", width="large"),
            },
        )


import hashlib

import streamlit as st
import pandas as pd
from pathlib import Path

from config import MAX_REVIEWS, MissingAPIKeyError, has_api_key
from data_validator import REQUIRED_COLUMNS, validate_reviews_with_report
from batch_processor import merge_results, process_reviews
from dashboard import display_dashboard
from insights_generator import generate_weekly_insights
from output_safety import make_csv_safe, sanitize_markdown


SAMPLE_CSV_PATH = Path(__file__).parent / "sample_data" / "sample_reviews.csv"


def reset_analysis_state():
    """Clear results that belong to a previously uploaded file."""

    st.session_state.validated_reviews = None
    st.session_state.rejected_reviews = None
    st.session_state.analysed_reviews = None
    st.session_state.weekly_report = None
    st.session_state.upload_error = None


def friendly_error(action: str, error: Exception) -> str:
    """Turn an exception into a short message for the user."""

    # Our own validation messages are safe and helpful to show as-is
    if isinstance(error, MissingAPIKeyError) or type(error) is ValueError:
        return str(error)

    return (
        f"{action} failed ({type(error).__name__}). "
        "Please try again. If the problem continues, check your API key, "
        "model name and internet connection."
    )


# =====================================================
# APPLICATION CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI-Powered Customer Review Analyzer",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None
    reset_analysis_state()


# =====================================================
# APPLICATION HEADER
# =====================================================

st.title("📊 AI-Powered Customer Review Analyzer")

st.write(
    "Analyse customer reviews using AI, identify customer "
    "sentiment, detect urgent issues, and generate business insights."
)

st.divider()


# =====================================================
# CSV FILE UPLOAD
# =====================================================

st.sidebar.header("Upload Customer Reviews")

uploaded_file = st.sidebar.file_uploader(
    "Choose a customer review CSV file",
    type=["csv"]
)

st.sidebar.info(
    "Required columns: " + ", ".join(REQUIRED_COLUMNS)
)

if SAMPLE_CSV_PATH.exists():
    st.sidebar.download_button(
        label="Download sample CSV",
        data=SAMPLE_CSV_PATH.read_bytes(),
        file_name="sample_reviews.csv",
        mime="text/csv"
    )

st.sidebar.caption(
    f"Up to {MAX_REVIEWS} reviews are analysed per run. "
    "Review text is sent to OpenAI for analysis."
)

if not has_api_key():
    st.sidebar.warning(
        "OPENAI_API_KEY is not set. You can upload and validate a CSV, "
        "but AI analysis needs an API key in your .env file."
    )


# =====================================================
# CSV VALIDATION
# =====================================================

if uploaded_file is None:

    # File removed: forget results that belonged to it
    if st.session_state.upload_signature is not None:
        st.session_state.upload_signature = None
        reset_analysis_state()

    st.info(
        "Upload a customer review CSV from the sidebar to get started. "
        "Not sure about the format? Download the sample CSV."
    )

else:

    file_bytes = uploaded_file.getvalue()
    signature = hashlib.sha256(file_bytes).hexdigest()

    # Only parse and validate when a different file is uploaded.
    # This also clears results from the previous file.
    if signature != st.session_state.upload_signature:

        st.session_state.upload_signature = signature
        reset_analysis_state()

        try:
            uploaded_df = pd.read_csv(
                uploaded_file,
                dtype={"review_id": "string"}
            )

            valid_df, rejected_df = validate_reviews_with_report(
                uploaded_df
            )

            st.session_state.validated_reviews = valid_df
            st.session_state.rejected_reviews = rejected_df

        except Exception as e:
            st.session_state.upload_error = str(e)

    if st.session_state.upload_error:
        st.error(f"CSV validation failed: {st.session_state.upload_error}")

    validated_df = st.session_state.validated_reviews
    rejected_df = st.session_state.rejected_reviews

    if validated_df is not None:

        if validated_df.empty:
            st.error(
                "No valid customer reviews found in the uploaded CSV."
            )
        else:
            st.success(
                f"CSV uploaded successfully! "
                f"{len(validated_df)} valid reviews found."
            )

        if rejected_df is not None and not rejected_df.empty:

            st.warning(
                f"{len(rejected_df)} row(s) were skipped because of "
                "data problems."
            )

            with st.expander("Show skipped rows"):

                st.dataframe(
                    rejected_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    label="Download skipped rows (CSV)",
                    data=make_csv_safe(rejected_df).to_csv(index=False),
                    file_name="skipped_reviews.csv",
                    mime="text/csv"
                )

        if not validated_df.empty:

            st.subheader("Uploaded Customer Reviews")

            st.dataframe(
                validated_df,
                use_container_width=True
            )


# =====================================================
# AI REVIEW ANALYSIS
# =====================================================

if (
    st.session_state.validated_reviews is not None
    and not st.session_state.validated_reviews.empty
):

    st.divider()

    st.subheader("AI-Powered Review Analysis")

    reviews_to_analyse = st.session_state.validated_reviews

    if len(reviews_to_analyse) > MAX_REVIEWS:
        st.warning(
            f"The file has {len(reviews_to_analyse)} valid reviews. "
            f"Only the first {MAX_REVIEWS} will be analysed "
            "(change MAX_REVIEWS in .env to raise the limit)."
        )
        reviews_to_analyse = reviews_to_analyse.head(MAX_REVIEWS)

    if st.button("Analyse Customer Reviews", type="primary"):

        try:
            reviews = reviews_to_analyse.to_dict(orient="records")

            with st.spinner(
                f"AI is analysing {len(reviews)} customer reviews. "
                "Please wait..."
            ):

                results = process_reviews(reviews)

            # Combine original reviews with AI results (by review_id)
            analysed_df = merge_results(reviews_to_analyse, results)

            st.session_state.analysed_reviews = analysed_df
            st.session_state.weekly_report = None

            success_count = int(
                (analysed_df["analysis_status"] == "success").sum()
            )

            st.success(
                f"Successfully analysed {success_count} of "
                f"{len(analysed_df)} reviews!"
            )

        except Exception as e:
            st.error(friendly_error("Review analysis", e))

    # Offer to retry reviews whose analysis failed
    analysed = st.session_state.analysed_reviews

    if analysed is not None:

        failed_df = analysed[analysed["analysis_status"] == "failed"]

        if not failed_df.empty:

            st.warning(
                f"{len(failed_df)} review(s) could not be analysed. "
                "They are excluded from the dashboard and insights."
            )

            with st.expander("Show failed reviews"):
                st.dataframe(
                    failed_df[["review_id", "review_text", "analysis_error"]],
                    use_container_width=True,
                    hide_index=True
                )

            if st.button("Retry failed reviews"):

                try:
                    with st.spinner("Retrying failed reviews..."):
                        retry_results = process_reviews(
                            failed_df.to_dict(orient="records")
                        )

                    st.session_state.analysed_reviews = merge_results(
                        analysed, retry_results
                    )
                    st.session_state.weekly_report = None
                    st.rerun()

                except Exception as e:
                    st.error(friendly_error("Retry", e))


# =====================================================
# DISPLAY AI ANALYSIS RESULTS
# =====================================================

if st.session_state.analysed_reviews is not None:

    st.divider()

    st.subheader("Customer Review Analysis Results")

    st.dataframe(
        st.session_state.analysed_reviews,
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# ANALYTICS DASHBOARD
# =====================================================

if st.session_state.analysed_reviews is not None:

    st.divider()

    analysed = st.session_state.analysed_reviews

    # Only successfully analysed reviews have sentiment/category/urgency
    display_dashboard(
        analysed[analysed["analysis_status"] == "success"]
    )


# =====================================================
# WEEKLY BUSINESS INSIGHTS
# =====================================================

if st.session_state.analysed_reviews is not None:

    st.divider()

    st.subheader("Weekly Business Insights")

    if st.button("Generate Weekly Insights"):

        try:
            with st.spinner(
                "AI is preparing your weekly business insights..."
            ):

                report = generate_weekly_insights(
                    st.session_state.analysed_reviews
                )

                # Strip links/images: the report is built from
                # untrusted customer content
                st.session_state.weekly_report = sanitize_markdown(report)

            st.success("Weekly insights generated successfully!")

        except Exception as e:
            st.error(friendly_error("Insights generation", e))

    # Display generated report
    if st.session_state.weekly_report is not None:

        st.caption(
            "AI-generated report. Please verify key figures before "
            "sharing it."
        )

        st.markdown(
            st.session_state.weekly_report
        )


# =====================================================
# DOWNLOAD REPORTS
# =====================================================

if st.session_state.analysed_reviews is not None:

    st.divider()

    st.subheader("Download Analysis Reports")

    col1, col2 = st.columns(2)

    with col1:
        # Escape cells that spreadsheets would run as formulas
        csv_data = make_csv_safe(
            st.session_state.analysed_reviews
        ).to_csv(index=False)

        st.download_button(
            label="Download Analysed Reviews (CSV)",
            data=csv_data,
            file_name="customer_review_analysis.csv",
            mime="text/csv"
        )

    with col2:
        if st.session_state.weekly_report is not None:

            st.download_button(
                label="Download Weekly Insights Report",
                data=st.session_state.weekly_report,
                file_name="weekly_business_insights.md",
                mime="text/markdown"
            )
        else:
            st.info("Generate weekly insights to download the report.")


# =====================================================
# CUSTOMER REVIEW FILTERS
# =====================================================

if st.session_state.analysed_reviews is not None:

    st.divider()

    st.subheader("Filter Customer Reviews")

    df = st.session_state.analysed_reviews.copy()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_platforms = st.multiselect(
            "Platform",
            options=sorted(df["platform"].dropna().unique()),
            default=sorted(df["platform"].dropna().unique())
        )

    with col2:
        selected_sentiments = st.multiselect(
            "Sentiment",
            options=["Positive", "Negative", "Neutral"],
            default=["Positive", "Negative", "Neutral"]
        )

    with col3:
        selected_categories = st.multiselect(
            "Category",
            options=sorted(df["category"].dropna().unique()),
            default=sorted(df["category"].dropna().unique())
        )

    with col4:
        selected_urgencies = st.multiselect(
            "Urgency",
            options=["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"]
        )

    # Apply filters
    filtered_df = df[
        df["platform"].isin(selected_platforms)
        & df["sentiment"].isin(selected_sentiments)
        & df["category"].isin(selected_categories)
        & df["urgency"].isin(selected_urgencies)
    ]

    st.write(f"Matching reviews: {len(filtered_df)}")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# HIGH URGENCY COMPLAINTS
# =====================================================

if st.session_state.analysed_reviews is not None:

    st.divider()

    st.subheader("🚨 High Urgency Complaints")

    urgent_df = st.session_state.analysed_reviews[
        st.session_state.analysed_reviews["urgency"] == "HIGH"
    ]

    if not urgent_df.empty:

        st.warning(
            f"{len(urgent_df)} high urgency complaint(s) "
            "require human review."
        )

        st.caption(
            "Suggested replies are AI-generated. Review and edit them "
            "before sending to customers."
        )

        st.dataframe(
            urgent_df[
                [
                    "review_id",
                    "date",
                    "platform",
                    "rating",
                    "category",
                    "summary",
                    "suggested_reply"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.success(
            "No high urgency complaints found."
        )

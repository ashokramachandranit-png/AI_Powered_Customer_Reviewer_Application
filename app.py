
import streamlit as st
import pandas as pd
from pathlib import Path

from data_validator import validate_reviews
from batch_processor import process_reviews
from dashboard import display_dashboard
from insights_generator import generate_weekly_insights


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

if "analysed_reviews" not in st.session_state:
    st.session_state.analysed_reviews = None

if "weekly_report" not in st.session_state:
    st.session_state.weekly_report = None


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
    "Required columns: review_id, date, platform, rating, review_text"
)


# =====================================================
# CSV VALIDATION
# =====================================================

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        validated_df = validate_reviews(uploaded_df)

        st.success(
            f"CSV uploaded successfully! "
            f"{len(validated_df)} valid reviews found."
        )

        st.subheader("Uploaded Customer Reviews")

        st.dataframe(
            validated_df,
            use_container_width=True
        )

        # Store validated reviews
        st.session_state.validated_reviews = validated_df

    except Exception as e:
        st.error(f"CSV validation failed: {e}")


# =====================================================
# AI REVIEW ANALYSIS
# =====================================================

if "validated_reviews" in st.session_state:

    st.divider()

    st.subheader("AI-Powered Review Analysis")

    if st.button("Analyse Customer Reviews", type="primary"):

        try:
            reviews = st.session_state.validated_reviews.to_dict(
                orient="records"
            )

            with st.spinner(
                "AI is analysing customer reviews. Please wait..."
            ):

                results = process_reviews(
                    reviews,
                    max_concurrency=5
                )

            # Combine original reviews with AI results
            analysed_df = pd.DataFrame(
                [
                    {**reviews[i], **results[i]}
                    for i in range(len(reviews))
                ]
            )

            st.session_state.analysed_reviews = analysed_df

            st.success(
                f"Successfully analysed {len(analysed_df)} reviews!"
            )

        except Exception as e:
            st.error(f"Review analysis failed: {e}")


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

    display_dashboard(
        st.session_state.analysed_reviews
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

                st.session_state.weekly_report = report

            st.success("Weekly insights generated successfully!")

        except Exception as e:
            st.error(f"Insights generation failed: {e}")

    # Display generated report
    if st.session_state.weekly_report is not None:

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
        csv_data = st.session_state.analysed_reviews.to_csv(
            index=False
        )

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

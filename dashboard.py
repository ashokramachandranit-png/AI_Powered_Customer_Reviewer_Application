
import streamlit as st
import pandas as pd
import plotly.express as px


def display_summary_cards(df: pd.DataFrame):
    """
    Display key customer review statistics.
    """

    total_reviews = len(df)

    positive_reviews = (df["sentiment"] == "Positive").sum()
    negative_reviews = (df["sentiment"] == "Negative").sum()
    neutral_reviews = (df["sentiment"] == "Neutral").sum()

    average_rating = pd.to_numeric(
        df["rating"], errors="coerce"
    ).mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Reviews", total_reviews)
    col2.metric("Positive Reviews", positive_reviews)
    col3.metric("Negative Reviews", negative_reviews)
    col4.metric("Neutral Reviews", neutral_reviews)
    col5.metric(
        "Average Rating",
        f"{average_rating:.2f}/5" if pd.notna(average_rating) else "N/A"
    )

def display_sentiment_chart(df: pd.DataFrame):
    """
    Display customer sentiment distribution.
    """

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        title="Customer Sentiment Distribution",
        hole=0.4
    )

    fig.update_traces(textinfo="percent+label")

    st.plotly_chart(fig, use_container_width=True)

def display_category_chart(df: pd.DataFrame):
    """
    Display customer review distribution by category.
    """

    category_counts = (
        df["category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = ["Category", "Count"]

    fig = px.bar(
        category_counts,
        x="Category",
        y="Count",
        title="Customer Reviews by Category",
        text="Count"
    )

    fig.update_layout(
        xaxis_title="Review Category",
        yaxis_title="Number of Reviews",
        xaxis_tickangle=-35
    )

    st.plotly_chart(fig, use_container_width=True)


def display_urgency_chart(df: pd.DataFrame):
    """
    Display customer review distribution by urgency.
    """

    urgency_counts = (
        df["urgency"]
        .value_counts()
        .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
        .reset_index()
    )

    urgency_counts.columns = ["Urgency", "Count"]

    fig = px.bar(
        urgency_counts,
        x="Urgency",
        y="Count",
        title="Customer Review Urgency Distribution",
        text="Count",
        category_orders={
            "Urgency": ["HIGH", "MEDIUM", "LOW"]
        }
    )

    fig.update_layout(
        xaxis_title="Urgency Level",
        yaxis_title="Number of Reviews"
    )

    st.plotly_chart(fig, use_container_width=True)


def display_platform_chart(df: pd.DataFrame):
    """
    Display customer review distribution by platform.
    """

    platform_counts = (
        df["platform"]
        .value_counts()
        .reset_index()
    )

    platform_counts.columns = ["Platform", "Count"]

    fig = px.bar(
        platform_counts,
        x="Platform",
        y="Count",
        title="Customer Reviews by Platform",
        text="Count"
    )

    fig.update_layout(
        xaxis_title="Platform",
        yaxis_title="Number of Reviews"
    )

    st.plotly_chart(fig, use_container_width=True)


def display_dashboard(df: pd.DataFrame):
    """
    Display the complete customer review analytics dashboard.
    """

    st.title("Customer Review Analytics Dashboard")

    if df.empty:
        st.warning("No review data available. Please analyse customer reviews first.")
        return

    # Display summary cards
    st.subheader("Overall Review Summary")
    display_summary_cards(df)

    st.divider()

    # Display sentiment and category charts
    col1, col2 = st.columns(2)

    with col1:
        display_sentiment_chart(df)

    with col2:
        display_category_chart(df)

    st.divider()

    # Display urgency and platform charts
    col3, col4 = st.columns(2)

    with col3:
        display_urgency_chart(df)

    with col4:
        display_platform_chart(df)

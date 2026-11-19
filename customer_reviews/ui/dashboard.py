"""Dashboard tab: summary metrics and charts."""

import pandas as pd
import streamlit as st

from customer_reviews.ui import charts


def display_summary_cards(df: pd.DataFrame) -> None:
    """Display key customer review statistics."""

    average_rating = pd.to_numeric(df["rating"], errors="coerce").mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total reviews", len(df))
    col2.metric("Positive", int((df["sentiment"] == "Positive").sum()))
    col3.metric("Negative", int((df["sentiment"] == "Negative").sum()))
    col4.metric("High urgency", int((df["urgency"] == "HIGH").sum()))
    col5.metric(
        "Average rating",
        f"{average_rating:.2f} / 5" if pd.notna(average_rating) else "N/A",
    )


def _chart(fig, key: str) -> None:
    st.plotly_chart(fig, key=key, theme="streamlit")


def render_dashboard_tab(
    df: pd.DataFrame,
    total_reviews: int,
    filters_active: bool,
    theme: str,
) -> None:
    """Display the analytics dashboard for the (filtered) analysed reviews."""

    if filters_active:
        st.caption(f"Showing {len(df)} of {total_reviews} analysed reviews (filters active).")

    if df.empty:
        st.info("No reviews match the current filters.")
        return

    display_summary_cards(df)

    col1, col2 = st.columns(2)
    with col1:
        _chart(charts.sentiment_donut(df, theme), "chart_sentiment")
    with col2:
        _chart(charts.category_bar(df, theme), "chart_category")

    col3, col4 = st.columns(2)
    with col3:
        _chart(charts.urgency_bar(df, theme), "chart_urgency")
    with col4:
        _chart(charts.platform_bar(df, theme), "chart_platform")

    col5, col6 = st.columns(2)
    with col5:
        _chart(charts.weekly_sentiment_bar(df, theme), "chart_weekly_sentiment")
    with col6:
        _chart(charts.weekly_rating_line(df, theme), "chart_weekly_rating")

    if {"aspects", "language"} <= set(df.columns):
        col7, col8 = st.columns(2)
        with col7:
            _chart(charts.aspects_bar(df, theme), "chart_aspects")
        with col8:
            _chart(charts.language_bar(df, theme), "chart_language")

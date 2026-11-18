from datetime import date

import pandas as pd
import pytest

from customer_reviews.services.filters import ReviewFilters, apply_filters, successful_reviews
from customer_reviews.ui import charts


@pytest.fixture
def analysed():
    return pd.DataFrame(
        [
            {"review_id": "R1", "date": pd.Timestamp("2026-10-05"), "platform": "Google",
             "rating": 5, "sentiment": "Positive", "category": "Food Quality",
             "urgency": "LOW", "summary": "Loved the curry", "review_text": "Great curry",
             "analysis_status": "success"},
            {"review_id": "R2", "date": pd.Timestamp("2026-10-13"), "platform": "Zomato",
             "rating": 1, "sentiment": "Negative", "category": "Refund",
             "urgency": "HIGH", "summary": "Charged twice", "review_text": "Refund please",
             "analysis_status": "success"},
            {"review_id": "R3", "date": pd.Timestamp("2026-10-20"), "platform": "Zomato",
             "rating": 3, "sentiment": "Neutral", "category": "Delivery",
             "urgency": "MEDIUM", "summary": "Okay", "review_text": "Cold food",
             "analysis_status": "success"},
            {"review_id": "R4", "date": pd.Timestamp("2026-10-21"), "platform": "Google",
             "rating": 2, "sentiment": None, "category": None, "urgency": None,
             "summary": None, "review_text": "x", "analysis_status": "failed"},
        ]
    )


def ids(df):
    return list(df["review_id"])


def test_successful_reviews(analysed):
    assert ids(successful_reviews(analysed)) == ["R1", "R2", "R3"]


def test_no_filters_returns_everything(analysed):
    filters = ReviewFilters()

    assert not filters.is_active()
    assert ids(apply_filters(analysed, filters)) == ["R1", "R2", "R3", "R4"]


def test_column_filters_combine(analysed):
    filters = ReviewFilters(platforms=["Zomato"], urgencies=["HIGH", "LOW"])

    assert filters.is_active()
    assert ids(apply_filters(analysed, filters)) == ["R2"]


def test_date_range_filter(analysed):
    filters = ReviewFilters(date_from=date(2026, 10, 10), date_to=date(2026, 10, 20))

    assert ids(apply_filters(analysed, filters)) == ["R2", "R3"]


def test_text_search_is_case_insensitive_and_literal(analysed):
    assert ids(apply_filters(analysed, ReviewFilters(search="CHARGED"))) == ["R2"]
    assert ids(apply_filters(analysed, ReviewFilters(search="cold"))) == ["R3"]
    # Regex characters are treated literally
    assert ids(apply_filters(analysed, ReviewFilters(search=".*"))) == []


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize(
    "builder",
    [
        charts.sentiment_donut, charts.category_bar, charts.urgency_bar,
        charts.platform_bar, charts.weekly_sentiment_bar, charts.weekly_rating_line,
    ],
)
def test_charts_build(analysed, builder, theme):
    fig = builder(successful_reviews(analysed), theme)

    assert fig.layout.title.text
    assert len(fig.data) >= 1


def test_sentiment_colours_are_semantic(analysed):
    fig = charts.sentiment_donut(successful_reviews(analysed), "light")
    colours = dict(zip(fig.data[0].labels, fig.data[0].marker.colors, strict=True))

    assert colours == charts.PALETTES["light"]["sentiment"]


def test_urgency_bars_keep_order_and_status_colours(analysed):
    fig = charts.urgency_bar(successful_reviews(analysed), "dark")

    assert [trace.name for trace in fig.data] == ["HIGH", "MEDIUM", "LOW"]
    assert fig.data[0].marker.color == charts.PALETTES["dark"]["urgency"]["HIGH"]


def test_bar_labels_are_not_clipped(analysed):
    fig = charts.platform_bar(successful_reviews(analysed), "light")

    assert fig.data[0].cliponaxis is False
    assert fig.layout.yaxis.range[1] > 2  # headroom above the tallest bar (2)

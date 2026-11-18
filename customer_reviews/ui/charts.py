"""
Plotly chart builders.

Pure functions (DataFrame in, figure out) so they can be tested without
Streamlit. Colours are semantic and validated for colour-vision
deficiencies in both light and dark themes:

- Sentiment is a polarity: blue (positive) / neutral gray / red (negative),
  never red/green.
- Urgency uses reserved status colours; bars are always labelled.
- Single-series charts (category, platform, rating) use one colour.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
URGENCY_ORDER = ["HIGH", "MEDIUM", "LOW"]

PALETTES = {
    "light": {
        "series": "#2a78d6",
        "sentiment": {"Positive": "#2a78d6", "Neutral": "#898781", "Negative": "#e34948"},
        "urgency": {"HIGH": "#d03b3b", "MEDIUM": "#fab219", "LOW": "#898781"},
        "gap": "#ffffff",        # page background, used for gaps between marks
        "grid": "#e1e0d9",
    },
    "dark": {
        "series": "#3987e5",
        "sentiment": {"Positive": "#3987e5", "Neutral": "#b0afa8", "Negative": "#e66767"},
        "urgency": {"HIGH": "#d03b3b", "MEDIUM": "#fab219", "LOW": "#b0afa8"},
        "gap": "#171614",
        "grid": "#2c2c2a",
    },
}


def _palette(theme: str) -> dict:
    return PALETTES["dark" if theme == "dark" else "light"]


def _style(fig: go.Figure, theme: str, title: str) -> go.Figure:
    """Shared layout: title, recessive grid, room for outside labels."""

    palette = _palette(theme)

    fig.update_layout(
        title=title,
        margin=dict(t=60, b=10, l=10, r=10),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=palette["grid"], zeroline=False)

    return fig


def _counts(df: pd.DataFrame, column: str, order: list[str] | None = None) -> pd.DataFrame:
    counts = df[column].value_counts()

    if order is not None:
        counts = counts.reindex(order, fill_value=0)

    return counts.rename_axis(column).reset_index(name="count")


def _add_outside_labels(fig: go.Figure, max_value: float, horizontal: bool = False) -> None:
    """Show values outside bars without clipping the tallest bar's label."""

    fig.update_traces(textposition="outside", cliponaxis=False)

    headroom = [0, max(max_value, 1) * 1.18]

    if horizontal:
        fig.update_xaxes(range=headroom)
    else:
        fig.update_yaxes(range=headroom)


def sentiment_donut(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    data = _counts(df, "sentiment", SENTIMENT_ORDER)
    data = data[data["count"] > 0]
    palette = _palette(theme)

    fig = px.pie(
        data,
        names="sentiment",
        values="count",
        hole=0.55,
        color="sentiment",
        color_discrete_map=palette["sentiment"],
        category_orders={"sentiment": SENTIMENT_ORDER},
    )
    fig.update_traces(
        textinfo="percent+label",
        sort=False,
        marker=dict(line=dict(color=palette["gap"], width=2)),
        hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>",
    )

    return _style(fig, theme, "Customer sentiment")


def category_bar(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    # Horizontal bars keep long category names readable
    data = _counts(df, "category").sort_values("count")

    fig = px.bar(data, x="count", y="category", orientation="h", text="count")
    fig.update_traces(
        marker_color=_palette(theme)["series"],
        hovertemplate="%{y}: %{x} reviews<extra></extra>",
    )
    _add_outside_labels(fig, data["count"].max() if not data.empty else 0, horizontal=True)
    fig.update_layout(xaxis_title="Reviews", yaxis_title=None)

    fig = _style(fig, theme, "Reviews by category")
    fig.update_xaxes(showgrid=True, gridcolor=_palette(theme)["grid"])
    fig.update_yaxes(showgrid=False)

    return fig


def urgency_bar(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    data = _counts(df, "urgency", URGENCY_ORDER)

    fig = px.bar(
        data,
        x="urgency",
        y="count",
        text="count",
        color="urgency",
        color_discrete_map=_palette(theme)["urgency"],
        category_orders={"urgency": URGENCY_ORDER},
    )
    fig.update_traces(hovertemplate="%{x}: %{y} reviews<extra></extra>")
    _add_outside_labels(fig, data["count"].max())
    fig.update_layout(showlegend=False, xaxis_title="Urgency", yaxis_title="Reviews")

    return _style(fig, theme, "Urgency")


def platform_bar(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    data = _counts(df, "platform").sort_values("count", ascending=False)

    fig = px.bar(data, x="platform", y="count", text="count")
    fig.update_traces(
        marker_color=_palette(theme)["series"],
        hovertemplate="%{x}: %{y} reviews<extra></extra>",
    )
    _add_outside_labels(fig, data["count"].max() if not data.empty else 0)
    fig.update_layout(xaxis_title="Platform", yaxis_title="Reviews")

    return _style(fig, theme, "Reviews by platform")


def _weekly(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    data["week"] = data["date"].dt.to_period("W").dt.start_time

    return data


def weekly_sentiment_bar(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    data = _weekly(df)
    palette = _palette(theme)

    counts = (
        data.groupby(["week", "sentiment"]).size().reset_index(name="count")
    )

    fig = px.bar(
        counts,
        x="week",
        y="count",
        color="sentiment",
        color_discrete_map=palette["sentiment"],
        category_orders={"sentiment": SENTIMENT_ORDER},
    )
    fig.update_traces(
        marker_line_color=palette["gap"],
        marker_line_width=2,
        hovertemplate="Week of %{x|%d %b %Y}<br>%{fullData.name}: %{y}<extra></extra>",
    )
    fig.update_layout(
        barmode="stack",
        xaxis_title="Week starting",
        yaxis_title="Reviews",
        legend_title_text=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.update_xaxes(tickformat="%d %b", dtick=7 * 24 * 60 * 60 * 1000)

    return _style(fig, theme, "Weekly sentiment")


def weekly_rating_line(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    data = _weekly(df)
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")

    weekly = data.groupby("week")["rating"].mean().round(2).reset_index()

    fig = px.line(weekly, x="week", y="rating", markers=True)
    fig.update_traces(
        line=dict(color=_palette(theme)["series"], width=2),
        marker=dict(size=8),
        hovertemplate="Week of %{x|%d %b %Y}<br>Average rating: %{y:.2f}<extra></extra>",
    )
    fig.update_layout(xaxis_title="Week starting", yaxis_title="Average rating")
    fig.update_yaxes(range=[0.8, 5.2], dtick=1)
    fig.update_xaxes(tickformat="%d %b", dtick=7 * 24 * 60 * 60 * 1000)

    return _style(fig, theme, "Average rating by week")

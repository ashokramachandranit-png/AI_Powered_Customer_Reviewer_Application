"""Filtering of analysed reviews (used by the sidebar filters)."""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class ReviewFilters:
    """Filter selections. Empty lists / None mean "no filter"."""

    platforms: list[str] = field(default_factory=list)
    sentiments: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    urgencies: list[str] = field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None
    search: str = ""

    def is_active(self) -> bool:
        return any(
            [
                self.platforms, self.sentiments, self.categories, self.urgencies,
                self.date_from, self.date_to, self.search.strip(),
            ]
        )


def successful_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Reviews the AI analysed successfully."""

    if "analysis_status" not in df.columns:
        return df

    return df[df["analysis_status"] == "success"]


def apply_filters(df: pd.DataFrame, filters: ReviewFilters) -> pd.DataFrame:
    """Return the reviews matching every active filter."""

    mask = pd.Series(True, index=df.index)

    for column, selected in [
        ("platform", filters.platforms),
        ("sentiment", filters.sentiments),
        ("category", filters.categories),
        ("urgency", filters.urgencies),
    ]:
        if selected:
            mask &= df[column].isin(selected)

    if filters.date_from or filters.date_to:
        dates = pd.to_datetime(df["date"], errors="coerce").dt.date

        if filters.date_from:
            mask &= dates >= filters.date_from
        if filters.date_to:
            mask &= dates <= filters.date_to

    search = filters.search.strip()

    if search:
        text_match = pd.Series(False, index=df.index)

        for column in ["review_text", "summary", "review_id"]:
            if column in df.columns:
                text_match |= (
                    df[column].astype(str).str.contains(search, case=False, regex=False, na=False)
                )

        mask &= text_match

    return df[mask]

import json
import math
import re
from functools import lru_cache

import pandas as pd

from customer_reviews.config import get_settings
from customer_reviews.llm import create_chat_model
from customer_reviews.prompts.insights import weekly_insights_prompt


# Limit how many urgent complaints are sent to the LLM (prompt size / cost)
MAX_URGENT_REVIEWS_IN_PROMPT = 25

# Matches the <data> / </data> tags used in the prompt (any case/spacing)
_DATA_TAG_PATTERN = re.compile(r"<\s*/?\s*data\s*>", re.IGNORECASE)


@lru_cache(maxsize=1)
def get_weekly_insights_chain():
    """Build the weekly insights chain on first use."""

    return weekly_insights_prompt | create_chat_model(get_settings())


def _successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only reviews that the AI analysed successfully."""

    if "analysis_status" in df.columns:
        df = df[df["analysis_status"] == "success"]

    return df


def _round_or_none(value) -> float | None:
    """Round a number to 2 decimals; return None for missing values."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    return round(float(value), 2)


def _optional_distributions(data: pd.DataFrame) -> dict:
    """Language and topic counts, when the analysis includes them."""

    result = {}

    if "language" in data.columns:
        result["language_distribution"] = data["language"].dropna().value_counts().to_dict()

    if "aspects" in data.columns:
        topics = data["aspects"].dropna().str.split("; ").explode()
        result["topics_mentioned"] = topics[topics != ""].value_counts().to_dict()

    return result


def build_insights_payload(df: pd.DataFrame) -> dict:
    """
    Aggregate analysed reviews into compact statistics for the LLM.
    """

    data = _successful_rows(df).copy()

    if data.empty:
        raise ValueError("No successfully analysed reviews available for insights.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")

    dated = data.dropna(subset=["date"])

    # Weekly volume and rating trend
    weekly_trends = (
        dated.groupby(dated["date"].dt.to_period("W"))
        .agg(
            total_reviews=("review_id", "count"),
            average_rating=("rating", "mean"),
            negative_reviews=("sentiment", lambda s: int((s == "Negative").sum())),
        )
        .reset_index()
    )
    weekly_trends["date"] = weekly_trends["date"].astype(str)
    weekly_trends["average_rating"] = weekly_trends["average_rating"].map(_round_or_none)

    # Urgent complaints (capped)
    urgent = data[data["urgency"] == "HIGH"]
    urgent_reviews = (
        urgent[["review_id", "platform", "category", "summary"]]
        .head(MAX_URGENT_REVIEWS_IN_PROMPT)
        .to_dict(orient="records")
    )

    return {
        "period_start": dated["date"].min().strftime("%Y-%m-%d") if not dated.empty else None,
        "period_end": dated["date"].max().strftime("%Y-%m-%d") if not dated.empty else None,
        "total_reviews": len(data),
        "average_rating": _round_or_none(data["rating"].mean()),
        "sentiment_distribution": data["sentiment"].value_counts().to_dict(),
        "category_distribution": data["category"].value_counts().to_dict(),
        "urgency_distribution": data["urgency"].value_counts().to_dict(),
        "platform_distribution": data["platform"].value_counts().to_dict(),
        **_optional_distributions(data),
        "weekly_trends": weekly_trends.to_dict(orient="records"),
        "high_urgency_review_count": len(urgent),
        "high_urgency_reviews": urgent_reviews,
    }


def generate_weekly_insights(df: pd.DataFrame, chain=None) -> str:
    """
    Generate a weekly business insights report.

    `chain` can be supplied for testing; by default the OpenAI chain is used.
    """

    if df.empty:
        raise ValueError("No review data available for insights.")

    payload = build_insights_payload(df)

    weekly_data = json.dumps(payload, indent=2, default=str)

    # Customer-derived summaries must not be able to close the <data> block
    weekly_data = _DATA_TAG_PATTERN.sub("", weekly_data)

    chain = chain or get_weekly_insights_chain()

    response = chain.invoke({"weekly_data": weekly_data})

    return _response_text(response)


def _response_text(response) -> str:
    """Extract plain text from a chat model response."""

    # `.text` joins text blocks, so it also works for providers that return
    # a list of content blocks (e.g. Claude with thinking).
    text = getattr(response, "text", None)

    return text if isinstance(text, str) else str(response.content)

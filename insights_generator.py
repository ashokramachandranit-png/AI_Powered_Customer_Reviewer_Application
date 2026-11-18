
import json
import pandas as pd

from config import get_llm
from prompts.weekly_insights_prompt import weekly_insights_prompt


# Create OpenAI model
llm = get_llm()

# Build weekly insights chain
weekly_insights_chain = weekly_insights_prompt | llm


def generate_weekly_insights(df: pd.DataFrame) -> str:
    """
    Generate a weekly business insights report.
    """

    if df.empty:
        raise ValueError("No review data available for insights.")

    # Convert date column
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Overall statistics
    total_reviews = len(data)

    sentiment_counts = (
        data["sentiment"].value_counts().to_dict()
    )

    category_counts = (
        data["category"].value_counts().to_dict()
    )

    urgency_counts = (
        data["urgency"].value_counts().to_dict()
    )

    platform_counts = (
        data["platform"].value_counts().to_dict()
    )

    average_rating = round(
        pd.to_numeric(data["rating"], errors="coerce").mean(), 2
    )

    # Weekly sentiment and category statistics
    weekly_trends = (
        data.groupby(data["date"].dt.to_period("W"))
        .agg(
            total_reviews=("review_id", "count"),
            average_rating=("rating", "mean")
        )
        .round(2)
        .reset_index()
    )

    # Urgent complaints
    urgent_reviews = data[
        data["urgency"] == "HIGH"
    ][
        ["review_id", "platform", "category", "summary"]
    ].to_dict(orient="records")

    # Prepare compact aggregated data
    weekly_data = {
        "total_reviews": total_reviews,
        "average_rating": average_rating,
        "sentiment_distribution": sentiment_counts,
        "category_distribution": category_counts,
        "urgency_distribution": urgency_counts,
        "platform_distribution": platform_counts,
        "weekly_trends": weekly_trends.to_dict(orient="records"),
        "high_urgency_reviews": urgent_reviews,
    }

    # Generate AI insights
    response = weekly_insights_chain.invoke(
        {
            "weekly_data": json.dumps(
                weekly_data,
                indent=2,
                default=str
            )
        }
    )

    return response.content

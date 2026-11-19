"""Structured output returned by the LLM for each review."""

from typing import Literal, get_args

from pydantic import BaseModel, Field


Category = Literal[
    "Food Quality",
    "Delivery",
    "Staff Behaviour",
    "Order Accuracy",
    "Pricing",
    "Refund",
    "Product Quality",
    "Packaging",
    "Customer Service",
    "Other",
]

CATEGORIES: list[str] = list(get_args(Category))

MAX_ASPECTS = 4


class ReviewAnalysis(BaseModel):
    """Structured analysis result for one customer review."""

    language: str = Field(
        description="Language the review is written in, as an English name, e.g. 'Spanish'."
    )

    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Overall sentiment expressed in the customer review."
    )

    category: Category = Field(
        description="Main category of the customer complaint or feedback."
    )

    aspects: list[Category] = Field(
        description=(
            "Every topic the review mentions, including the main category, "
            f"most important first (at most {MAX_ASPECTS})."
        )
    )

    urgency: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Priority level based on the seriousness of the issue."
    )

    summary: str = Field(
        description="Short summary of the customer review in English."
    )

    suggested_reply: str = Field(
        description="A polite and professional suggested response in the review's language."
    )

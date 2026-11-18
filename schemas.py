
from typing import Literal
from pydantic import BaseModel, Field


class ReviewAnalysis(BaseModel):
    """Structured analysis result for one customer review."""

    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Overall sentiment expressed in the customer review."
    )

    category: Literal[
        "Food Quality",
        "Delivery",
        "Staff Behaviour",
        "Order Accuracy",
        "Pricing",
        "Refund",
        "Product Quality",
        "Packaging",
        "Customer Service",
        "Other"
    ] = Field(
        description="Main category of the customer complaint or feedback."
    )

    urgency: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Priority level based on the seriousness of the issue."
    )

    summary: str = Field(
        description="Short summary of the customer review in English."
    )

    suggested_reply: str = Field(
        description="A polite and professional suggested response in English."
    )

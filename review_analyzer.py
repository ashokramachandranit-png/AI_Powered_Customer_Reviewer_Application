import re
from functools import lru_cache

from config import MAX_RETRIES, MAX_REVIEW_CHARS, get_llm
from schemas import ReviewAnalysis
from prompts.review_analysis_prompt import review_analysis_prompt


# Matches the <review> / </review> tags used in the prompt (any case/spacing)
_DELIMITER_TAG_PATTERN = re.compile(r"<\s*/?\s*review\s*>", re.IGNORECASE)


def sanitize_review_text(text: str) -> str:
    """
    Prepare untrusted review text for the prompt.

    Removes the prompt delimiter tags (so a review cannot "close" its
    own block and inject instructions) and limits the length to control
    cost.
    """

    text = _DELIMITER_TAG_PATTERN.sub("", str(text)).strip()

    if len(text) > MAX_REVIEW_CHARS:
        text = text[:MAX_REVIEW_CHARS] + " [truncated]"

    return text


def build_review_input(review: dict) -> dict:
    """Build the prompt variables for one customer review."""

    date = review.get("date")

    # Show dates as YYYY-MM-DD instead of a full timestamp
    if hasattr(date, "strftime"):
        date = date.strftime("%Y-%m-%d")

    return {
        "review_id": review["review_id"],
        "date": date,
        "platform": review["platform"],
        "rating": review["rating"],
        "review_text": sanitize_review_text(review["review_text"]),
    }


@lru_cache(maxsize=1)
def get_review_analysis_chain():
    """
    Build the LangChain review analysis chain on first use.

    Building lazily (instead of at import time) lets the app start and
    show a helpful message when the API key is missing.
    """

    llm = get_llm()

    # Enable structured output using the Pydantic schema
    structured_llm = llm.with_structured_output(ReviewAnalysis)

    # Retry transient failures (rate limits, timeouts, invalid output)
    return (review_analysis_prompt | structured_llm).with_retry(
        stop_after_attempt=MAX_RETRIES,
        wait_exponential_jitter=True,
    )


def analyze_review(review: dict) -> dict:
    """
    Analyse one customer review and return structured results.
    """

    result = get_review_analysis_chain().invoke(
        build_review_input(review)
    )

    return result.model_dump()

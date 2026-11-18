import pandas as pd

from config import MAX_CONCURRENCY
from review_analyzer import build_review_input, get_review_analysis_chain


# Columns added to every review by the AI analysis
ANALYSIS_COLUMNS = [
    "sentiment",
    "category",
    "urgency",
    "summary",
    "suggested_reply",
    "analysis_status",
    "analysis_error",
]


def _short_error(error: Exception) -> str:
    """Return a short, user-friendly error description."""

    message = str(error).strip().splitlines()[0] if str(error).strip() else ""

    return f"{type(error).__name__}: {message}"[:200]


def process_reviews(
    reviews: list[dict],
    max_concurrency: int = MAX_CONCURRENCY,
    chain=None,
) -> list[dict]:
    """
    Analyse multiple customer reviews using LangChain batch processing.

    One failed review never fails the whole batch: every review gets a
    result with `analysis_status` set to "success" or "failed".

    `chain` can be supplied for testing; by default the OpenAI chain is used.
    """

    if not reviews:
        return []

    chain = chain or get_review_analysis_chain()

    inputs = [build_review_input(review) for review in reviews]

    results = chain.batch(
        inputs,
        config={"max_concurrency": max_concurrency},
        return_exceptions=True,
    )

    output = []

    for review, result in zip(reviews, results, strict=True):

        if isinstance(result, Exception):
            output.append(
                {
                    "review_id": review["review_id"],
                    "sentiment": None,
                    "category": None,
                    "urgency": None,
                    "summary": None,
                    "suggested_reply": None,
                    "analysis_status": "failed",
                    "analysis_error": _short_error(result),
                }
            )
        else:
            output.append(
                {
                    "review_id": review["review_id"],
                    **result.model_dump(),
                    "analysis_status": "success",
                    "analysis_error": "",
                }
            )

    return output


def merge_results(
    reviews_df: pd.DataFrame,
    results: list[dict],
) -> pd.DataFrame:
    """
    Combine the original reviews with AI results, matched by review_id.

    Existing analysis columns in `reviews_df` are replaced for the reviews
    present in `results` and kept for all other reviews, so this also
    works for re-analysing only the failed reviews.
    """

    if not results:
        return reviews_df.copy()

    results_df = pd.DataFrame(results).set_index("review_id")

    merged = reviews_df.copy().set_index("review_id", drop=False)

    for column in ANALYSIS_COLUMNS:
        if column not in merged.columns:
            merged[column] = None

        merged[column] = merged[column].astype(object)

        matched = results_df.index.intersection(merged.index)
        merged.loc[matched, column] = results_df.loc[matched, column].astype(object)

    return merged.reset_index(drop=True)

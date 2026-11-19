"""
Review analysis: prompt preparation, the LLM chain, batch processing with
per-review error handling, result caching and progress reporting.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, MutableMapping
from functools import lru_cache

import pandas as pd

from customer_reviews.config import get_settings
from customer_reviews.llm import create_chat_model, with_structured_output
from customer_reviews.prompts import PROMPT_VERSION
from customer_reviews.prompts.review_analysis import (
    SYSTEM_MESSAGE,
    review_analysis_prompt,
)
from customer_reviews.schemas import MAX_ASPECTS, ReviewAnalysis

# Separator for the aspects column (a list stored as text, CSV-friendly)
ASPECT_SEPARATOR = "; "


# Columns added to every review by the AI analysis
ANALYSIS_COLUMNS = [
    "language",
    "sentiment",
    "category",
    "aspects",
    "urgency",
    "summary",
    "suggested_reply",
    "analysis_status",
    "analysis_error",
]

# Matches the <review> / </review> tags used in the prompt (any case/spacing)
_DELIMITER_TAG_PATTERN = re.compile(r"<\s*/?\s*review\s*>", re.IGNORECASE)

# Rough token estimates (about 4 characters per token)
CHARS_PER_TOKEN = 4
PROMPT_OVERHEAD_TOKENS = len(SYSTEM_MESSAGE) // CHARS_PER_TOKEN + 60
OUTPUT_TOKENS_PER_REVIEW = 200

ProgressCallback = Callable[[int, int], None]


# =====================================================
# PROMPT INPUTS
# =====================================================

def sanitize_review_text(text: str, max_chars: int | None = None) -> str:
    """
    Prepare untrusted review text for the prompt.

    Removes the prompt delimiter tags (so a review cannot "close" its own
    block and inject instructions) and limits the length to control cost.
    """

    if max_chars is None:
        max_chars = get_settings().max_review_chars

    text = _DELIMITER_TAG_PATTERN.sub("", str(text)).strip()

    if len(text) > max_chars:
        text = text[:max_chars] + " [truncated]"

    return text


def build_review_input(review: dict, max_chars: int | None = None) -> dict:
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
        "review_text": sanitize_review_text(review["review_text"], max_chars),
    }


# =====================================================
# LLM CHAIN
# =====================================================

@lru_cache(maxsize=1)
def get_review_analysis_chain():
    """
    Build the review analysis chain on first use.

    Building lazily (instead of at import time) lets the app start and show
    a helpful message when the API key is missing.
    """

    settings = get_settings()

    llm = create_chat_model(settings)
    structured_llm = with_structured_output(llm, ReviewAnalysis, settings)

    # Retry transient failures (rate limits, timeouts, invalid output)
    return (review_analysis_prompt | structured_llm).with_retry(
        stop_after_attempt=settings.max_retries,
        wait_exponential_jitter=True,
    )


# =====================================================
# CACHING
# =====================================================

def cache_key(prompt_input: dict, model_id: str) -> str:
    """Stable key for one review's analysis with a given model and prompt."""

    payload = json.dumps(
        {"model": model_id, "prompt": PROMPT_VERSION, "input": prompt_input},
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =====================================================
# BATCH PROCESSING
# =====================================================

def _short_error(error: Exception) -> str:
    """Return a short, user-friendly error description."""

    text = str(error).strip()
    message = text.splitlines()[0] if text else ""

    return f"{type(error).__name__}: {message}"[:200]


def _failed_row(review_id, error: Exception) -> dict:
    row = {column: None for column in ANALYSIS_COLUMNS}
    row.update(
        review_id=review_id,
        analysis_status="failed",
        analysis_error=_short_error(error),
    )
    return row


def _join_aspects(analysis: dict) -> str:
    """Main category first, no duplicates, at most MAX_ASPECTS."""

    aspects = [analysis.get("category"), *(analysis.get("aspects") or [])]
    unique = list(dict.fromkeys(a for a in aspects if a))

    return ASPECT_SEPARATOR.join(unique[:MAX_ASPECTS])


def _success_row(review_id, analysis: dict) -> dict:
    return {
        "review_id": review_id,
        **analysis,
        "aspects": _join_aspects(analysis),
        "analysis_status": "success",
        "analysis_error": "",
    }


def process_reviews(
    reviews: list[dict],
    *,
    chain=None,
    max_concurrency: int | None = None,
    cache: MutableMapping[str, dict] | None = None,
    model_id: str | None = None,
    max_chars: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[dict]:
    """
    Analyse customer reviews in parallel.

    - One failed review never fails the batch: every review gets a result
      with `analysis_status` "success" or "failed".
    - With a `cache`, reviews already analysed (same text, model and prompt
      version) are reused instead of calling the LLM again.
    - `on_progress(done, total)` is called as reviews complete.

    `chain` can be supplied for testing; by default the configured LLM is used.
    """

    if not reviews:
        return []

    if chain is None:
        chain = get_review_analysis_chain()

    if max_concurrency is None:
        max_concurrency = get_settings().max_concurrency

    if model_id is None:
        model_id = get_settings().model_id

    inputs = [build_review_input(review, max_chars) for review in reviews]
    keys = [cache_key(prompt_input, model_id) for prompt_input in inputs]

    results: list[dict | None] = [None] * len(reviews)
    pending: list[int] = []

    for i, key in enumerate(keys):
        if cache is not None and key in cache:
            results[i] = _success_row(reviews[i]["review_id"], cache[key])
        else:
            pending.append(i)

    total = len(reviews)
    done = total - len(pending)

    if on_progress:
        on_progress(done, total)

    if pending:
        # Each review is a separate `invoke` so the chain's retry policy
        # applies per review. (LangChain's batch_as_completed bypasses
        # .with_retry().) Results are handled here, in the calling thread,
        # so progress callbacks can safely update the UI.
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(chain.invoke, inputs[i]): i for i in pending
            }

            for future in as_completed(futures):
                i = futures[future]
                review_id = reviews[i]["review_id"]

                try:
                    analysis = future.result().model_dump()
                except Exception as error:
                    results[i] = _failed_row(review_id, error)
                else:
                    results[i] = _success_row(review_id, analysis)

                    if cache is not None:
                        cache[keys[i]] = analysis

                done += 1

                if on_progress:
                    on_progress(done, total)

    return results


def merge_results(reviews_df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    """
    Combine the original reviews with AI results, matched by review_id.

    Existing analysis columns in `reviews_df` are replaced for the reviews
    present in `results` and kept for all other reviews, so this also works
    for re-analysing only the failed reviews.
    """

    if not results:
        return reviews_df.copy()

    results_df = pd.DataFrame(results).set_index("review_id")

    merged = reviews_df.copy().set_index("review_id", drop=False)
    matched = results_df.index.intersection(merged.index)

    for column in ANALYSIS_COLUMNS:
        if column not in merged.columns:
            merged[column] = None

        merged[column] = merged[column].astype(object)
        merged.loc[matched, column] = results_df.loc[matched, column].astype(object)

    return merged.reset_index(drop=True)


# =====================================================
# USAGE ESTIMATE
# =====================================================

def estimate_usage(
    reviews: list[dict],
    *,
    cache: MutableMapping[str, dict] | None = None,
    model_id: str | None = None,
    max_chars: int | None = None,
) -> dict:
    """
    Roughly estimate LLM usage for analysing `reviews`.

    Reviews already in the cache cost nothing. Token counts are estimates
    (about 4 characters per token) and exclude any model "thinking" tokens.
    """

    if model_id is None:
        model_id = get_settings().model_id

    new_reviews = 0
    input_tokens = 0

    for review in reviews:
        prompt_input = build_review_input(review, max_chars)

        if cache is not None and cache_key(prompt_input, model_id) in cache:
            continue

        new_reviews += 1
        input_tokens += PROMPT_OVERHEAD_TOKENS + len(prompt_input["review_text"]) // CHARS_PER_TOKEN

    return {
        "total_reviews": len(reviews),
        "cached_reviews": len(reviews) - new_reviews,
        "new_reviews": new_reviews,
        "input_tokens": input_tokens,
        "output_tokens": new_reviews * OUTPUT_TOKENS_PER_REVIEW,
    }

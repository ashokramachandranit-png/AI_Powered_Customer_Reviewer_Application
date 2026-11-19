import pandas as pd
from langchain_core.runnables import RunnableLambda

from customer_reviews.services.analysis import (
    estimate_usage,
    merge_results,
    process_reviews,
)
from tests.conftest import make_analysis


REVIEWS = [
    {"review_id": "R1", "date": pd.Timestamp("2026-10-05"), "platform": "Google",
     "rating": 2, "review_text": "Late"},
    {"review_id": "R2", "date": pd.Timestamp("2026-10-06"), "platform": "Zomato",
     "rating": 1, "review_text": "FAIL please"},
    {"review_id": "R3", "date": pd.Timestamp("2026-10-07"), "platform": "Swiggy",
     "rating": 5, "review_text": "Great"},
]


def test_empty_input_returns_empty_list(fake_chain):
    assert process_reviews([], chain=fake_chain) == []


def test_one_failure_does_not_fail_the_batch(fake_chain):
    results = process_reviews(REVIEWS, chain=fake_chain)

    assert [r["review_id"] for r in results] == ["R1", "R2", "R3"]
    assert [r["analysis_status"] for r in results] == ["success", "failed", "success"]
    assert results[0]["summary"] == "Summary of R1"
    assert results[1]["sentiment"] is None
    assert "simulated rate limit" in results[1]["analysis_error"]


def test_prompt_inputs_are_prepared(fake_chain):
    process_reviews(REVIEWS[:1], chain=fake_chain)

    sent = fake_chain.calls[0]
    assert sent["date"] == "2026-10-05"
    assert sent["review_text"] == "Late"


def test_progress_is_reported(fake_chain):
    updates = []

    process_reviews(REVIEWS, chain=fake_chain, on_progress=lambda d, t: updates.append((d, t)))

    assert updates[0] == (0, 3)
    assert updates[-1] == (3, 3)
    assert len(updates) == 4


def test_cache_skips_already_analysed_reviews(fake_chain):
    cache = {}

    first = process_reviews(REVIEWS, chain=fake_chain, cache=cache, model_id="m1")
    calls_after_first = len(fake_chain.calls)
    second = process_reviews(REVIEWS, chain=fake_chain, cache=cache, model_id="m1")

    # Successful results are cached; only the failed review is sent again
    assert len(cache) == 2
    assert len(fake_chain.calls) == calls_after_first + 1
    assert second[0] == first[0]
    assert second[1]["analysis_status"] == "failed"


def test_cache_is_per_model(fake_chain):
    cache = {}

    process_reviews(REVIEWS[:1], chain=fake_chain, cache=cache, model_id="m1")
    process_reviews(REVIEWS[:1], chain=fake_chain, cache=cache, model_id="m2")

    assert len(fake_chain.calls) == 2


def test_retries_transient_errors_with_batch_as_completed():
    attempts = {"count": 0}

    def flaky(inputs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return make_analysis()

    chain = RunnableLambda(flaky).with_retry(stop_after_attempt=3, wait_exponential_jitter=False)

    results = process_reviews(REVIEWS[:1], chain=chain)

    assert results[0]["analysis_status"] == "success"
    assert attempts["count"] == 2


def test_estimate_usage_counts_only_uncached_reviews(fake_chain):
    cache = {}
    process_reviews(REVIEWS[:1], chain=fake_chain, cache=cache, model_id="m1")

    estimate = estimate_usage(REVIEWS, cache=cache, model_id="m1")

    assert estimate["total_reviews"] == 3
    assert estimate["cached_reviews"] == 1
    assert estimate["new_reviews"] == 2
    assert estimate["input_tokens"] > 0
    assert estimate["output_tokens"] == 2 * 200


def test_merge_results_matches_by_review_id(fake_chain):
    reviews_df = pd.DataFrame(REVIEWS)
    results = process_reviews(REVIEWS, chain=fake_chain)

    # Shuffle results: merging must not depend on order
    merged = merge_results(reviews_df, list(reversed(results)))

    assert list(merged["review_id"]) == ["R1", "R2", "R3"]
    assert merged.loc[0, "summary"] == "Summary of R1"
    assert merged.loc[2, "summary"] == "Summary of R3"
    assert merged.loc[1, "analysis_status"] == "failed"


def test_merge_results_updates_only_retried_reviews(fake_chain):
    reviews_df = pd.DataFrame(REVIEWS)
    merged = merge_results(reviews_df, process_reviews(REVIEWS, chain=fake_chain))

    # Retry R2 successfully
    fixed = dict(REVIEWS[1], review_text="Now it works")
    retried = merge_results(merged, process_reviews([fixed], chain=fake_chain))

    assert list(retried["analysis_status"]) == ["success"] * 3
    assert retried.loc[1, "summary"] == "Summary of R2"
    assert retried.loc[0, "summary"] == "Summary of R1"
    assert len(retried) == 3

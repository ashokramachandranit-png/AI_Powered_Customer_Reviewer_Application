import pandas as pd

from batch_processor import merge_results, process_reviews


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
    # Other rows are untouched
    assert retried.loc[0, "summary"] == "Summary of R1"
    assert len(retried) == 3

import pandas as pd
import pytest

from data_validator import validate_reviews, validate_reviews_with_report


def make_df(rows):
    return pd.DataFrame(
        rows, columns=["review_id", "date", "platform", "rating", "review_text"]
    )


def test_valid_rows_are_cleaned():
    df = make_df([[" R1 ", "2026-10-05", " Google ", "5", "  Great food  "]])

    valid, rejected = validate_reviews_with_report(df)

    assert rejected.empty
    assert valid.loc[0, "review_id"] == "R1"
    assert valid.loc[0, "platform"] == "Google"
    assert valid.loc[0, "review_text"] == "Great food"
    assert valid.loc[0, "rating"] == 5
    assert valid.loc[0, "date"] == pd.Timestamp("2026-10-05")


def test_missing_columns_raise():
    df = pd.DataFrame({"review_id": ["R1"], "review_text": ["hi"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_reviews_with_report(df)


def test_column_names_are_case_insensitive():
    df = pd.DataFrame(
        {
            "Review_ID": ["R1"],
            "Date": ["2026-10-05"],
            "Platform": ["Google"],
            " Rating ": [4],
            "Review_Text": ["Nice"],
        }
    )

    valid, _ = validate_reviews_with_report(df)

    assert len(valid) == 1


def test_rejected_rows_have_reasons():
    df = make_df(
        [
            ["R1", "2026-10-05", "Google", 5, "Good"],
            [None, "2026-10-05", "Google", 5, "No id"],
            ["R3", "2026-10-05", "Google", 5, "   "],
            ["R4", "2026-10-05", None, 5, "No platform"],
            ["R5", "2026-10-05", "Google", "abc", "Bad rating"],
            ["R6", "2026-10-05", "Google", 9, "Out of range"],
            ["R7", "not a date", "Google", 3, "Bad date"],
            ["R1", "2026-10-06", "Google", 4, "Duplicate"],
        ]
    )

    valid, rejected = validate_reviews_with_report(df)

    assert list(valid["review_id"]) == ["R1"]
    assert list(rejected["rejection_reason"]) == [
        "Missing review_id",
        "Missing review_text",
        "Missing platform",
        "Rating is not a number",
        "Rating must be between 1 and 5",
        "Missing or invalid date",
        "Duplicate review_id",
    ]
    # Spreadsheet row numbers (header is row 1)
    assert list(rejected["csv_row"]) == [3, 4, 5, 6, 7, 8, 9]
    # Rejected rows show the original values from the file
    assert rejected.loc[5, "date"] == "not a date"
    assert rejected.loc[3, "rating"] == "abc"


def test_invalid_row_does_not_block_valid_duplicate():
    # The first R1 is invalid, so the second R1 should be kept
    df = make_df(
        [
            ["R1", "2026-10-05", "Google", 9, "Invalid rating"],
            ["R1", "2026-10-06", "Google", 4, "Valid"],
        ]
    )

    valid, rejected = validate_reviews_with_report(df)

    assert list(valid["review_text"]) == ["Valid"]
    assert list(rejected["rejection_reason"]) == ["Rating must be between 1 and 5"]


def test_mixed_date_formats_are_parsed():
    df = make_df(
        [
            ["R1", "2026-10-05", "Google", 5, "a"],
            ["R2", "October 6, 2026", "Google", 5, "b"],
        ]
    )

    valid, rejected = validate_reviews_with_report(df)

    assert rejected.empty
    assert list(valid["date"]) == [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06")]


def test_validate_reviews_raises_when_nothing_is_valid():
    df = make_df([["R1", "2026-10-05", "Google", 0, "bad"]])

    with pytest.raises(ValueError, match="No valid customer reviews"):
        validate_reviews(df)


def test_sample_csv_is_fully_valid():
    df = pd.read_csv("sample_data/sample_reviews.csv", dtype={"review_id": "string"})

    valid, rejected = validate_reviews_with_report(df)

    assert rejected.empty
    assert len(valid) == len(df)

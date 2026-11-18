import pandas as pd


REQUIRED_COLUMNS = [
    "review_id",
    "date",
    "platform",
    "rating",
    "review_text",
]

TEXT_COLUMNS = ["review_id", "platform", "review_text"]


def validate_reviews_with_report(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate uploaded customer review data.

    Returns two DataFrames:
    - valid reviews, cleaned and ready for analysis
    - rejected rows, with a `rejection_reason` column explaining why

    Raises ValueError when required columns are missing.
    """

    # Accept column names with different case or surrounding spaces
    df = df.rename(columns=lambda c: str(c).strip().lower())

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only required columns
    data = df[REQUIRED_COLUMNS].copy()

    # Original row number as shown in a spreadsheet (header = row 1)
    data.insert(0, "csv_row", range(2, len(data) + 2))

    # Keep the original values to show users exactly what was rejected
    original = data.copy()

    # Clean text fields; empty text becomes missing
    for column in TEXT_COLUMNS:
        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # Parse rating and date
    raw_rating = data["rating"]
    data["rating"] = pd.to_numeric(raw_rating, errors="coerce")
    data["date"] = pd.to_datetime(
        data["date"], errors="coerce", format="mixed"
    )

    # Record the first problem found for each row
    reason = pd.Series("", index=data.index, dtype="object")

    def flag(mask: pd.Series, message: str) -> None:
        reason[mask & (reason == "")] = message

    flag(data["review_id"].isna(), "Missing review_id")
    flag(data["review_text"].isna(), "Missing review_text")
    flag(data["platform"].isna(), "Missing platform")
    flag(raw_rating.isna(), "Missing rating")
    flag(data["rating"].isna(), "Rating is not a number")
    flag(~data["rating"].between(1, 5), "Rating must be between 1 and 5")
    flag(data["date"].isna(), "Missing or invalid date")

    # Duplicate IDs among otherwise-valid rows (keep the first)
    candidates = reason == ""
    duplicated = pd.Series(False, index=data.index)
    duplicated[candidates] = (
        data.loc[candidates, "review_id"].duplicated(keep="first")
    )
    flag(duplicated, "Duplicate review_id")

    rejected = original[reason != ""].copy()
    rejected["rejection_reason"] = reason[reason != ""]

    valid = (
        data[reason == ""]
        .drop(columns=["csv_row"])
        .reset_index(drop=True)
    )

    # Use plain Python strings for the text columns
    for column in TEXT_COLUMNS:
        valid[column] = valid[column].astype(str)

    return valid, rejected.reset_index(drop=True)


def validate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate uploaded customer review data and return the valid reviews.
    """

    valid, _ = validate_reviews_with_report(df)

    if valid.empty:
        raise ValueError(
            "No valid customer reviews found in the uploaded CSV."
        )

    return valid

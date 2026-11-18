
import pandas as pd


REQUIRED_COLUMNS = [
    "review_id",
    "date",
    "platform",
    "rating",
    "review_text",
]


def validate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate uploaded customer review data.
    """

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only required columns
    df = df[REQUIRED_COLUMNS].copy()

    # Remove rows with missing essential values
    df = df.dropna(
        subset=["review_id", "platform", "rating", "review_text"]
    )

    # Clean text fields
    df["review_id"] = df["review_id"].astype(str).str.strip()
    df["platform"] = df["platform"].astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str).str.strip()

    # Convert rating to numeric
    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    )

    # Remove invalid ratings
    df = df[df["rating"].between(1, 5)]

    # Remove empty reviews
    df = df[df["review_text"] != ""]

    # Remove duplicate review IDs
    df = df.drop_duplicates(
        subset=["review_id"], keep="first"
    )

    # Convert date column
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    # Reset index
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid customer reviews found in the uploaded CSV."
        )

    return df

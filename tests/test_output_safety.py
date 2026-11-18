import pandas as pd

from customer_reviews.core.safety import make_csv_safe, sanitize_markdown


def test_formula_cells_are_escaped():
    df = pd.DataFrame(
        {
            "review_text": ["=HYPERLINK(\"http://evil\")", "+1", "@SUM(A1)", "-2", "Fine"],
            "rating": [1, 2, 3, 4, 5],
        }
    )

    safe = make_csv_safe(df)

    assert list(safe["review_text"]) == [
        "'=HYPERLINK(\"http://evil\")", "'+1", "'@SUM(A1)", "'-2", "Fine"
    ]
    # Numeric columns are unchanged and stay numeric
    assert list(safe["rating"]) == [1, 2, 3, 4, 5]
    # Original DataFrame is not modified
    assert df.loc[0, "review_text"].startswith("=")


def test_missing_values_are_kept():
    df = pd.DataFrame({"summary": [None, "ok"]})

    safe = make_csv_safe(df)

    assert pd.isna(safe.loc[0, "summary"])


def test_markdown_images_and_links_are_removed():
    text = (
        "## Report\n"
        "![tracker](https://evil.example/pixel.png)\n"
        "See [our portal](https://evil.example/login) or <https://evil.example>."
    )

    cleaned = sanitize_markdown(text)

    assert "evil.example" not in cleaned
    assert "tracker" in cleaned
    assert "our portal" in cleaned
    assert "## Report" in cleaned

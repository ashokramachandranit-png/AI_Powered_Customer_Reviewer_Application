import re

import pandas as pd


# Characters that make spreadsheet apps treat a cell as a formula
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# Markdown images: ![alt](url)  -> alt
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")

# Markdown links: [text](url)  -> text
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")

# Autolinks: <https://example.com>
_AUTOLINK = re.compile(r"<(?:https?|ftp|mailto):[^>]*>", re.IGNORECASE)


def escape_formula(value):
    """Prefix text that a spreadsheet would run as a formula."""

    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value

    return value


def make_csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Protect a DataFrame against CSV / formula injection before export.

    Customer reviews and AI output are untrusted. A cell such as
    =HYPERLINK(...) would run as a formula when the CSV is opened in
    Excel or Google Sheets. Only text columns are changed, so numeric
    values such as ratings stay numeric.
    """

    safe = df.copy()

    for column in safe.columns:
        if not pd.api.types.is_numeric_dtype(safe[column]) and not (
            pd.api.types.is_datetime64_any_dtype(safe[column])
        ):
            safe[column] = safe[column].map(escape_formula)

    return safe


def sanitize_markdown(text: str) -> str:
    """
    Remove images and links from AI-generated Markdown.

    The report is generated from untrusted review content, so it must not
    be able to load remote images (tracking) or show deceptive links.
    """

    text = _MARKDOWN_IMAGE.sub(r"\1", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _AUTOLINK.sub("", text)

    return text

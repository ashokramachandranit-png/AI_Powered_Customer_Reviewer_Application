"""
SQLite persistence: analysis cache, review history and reply workflow.

Uses only the standard library. A new connection is opened per operation,
which is cheap for SQLite and safe across Streamlit's script threads.
"""

import json
import sqlite3
from collections.abc import Iterator, MutableMapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPLY_STATUSES = ["Open", "Approved", "Sent", "Dismissed"]

REVIEW_COLUMNS = [
    "review_id", "date", "platform", "rating", "review_text",
    "language", "sentiment", "category", "aspects", "urgency",
    "summary", "suggested_reply", "model_id", "analysed_at",
]

REPLY_COLUMNS = ["reply_status", "reply_text", "reply_updated_at"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_cache (
    key         TEXT PRIMARY KEY,
    result      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id         TEXT PRIMARY KEY,
    date              TEXT,
    platform          TEXT,
    rating            REAL,
    review_text       TEXT,
    language          TEXT,
    sentiment         TEXT,
    category          TEXT,
    aspects           TEXT,
    urgency           TEXT,
    summary           TEXT,
    suggested_reply   TEXT,
    model_id          TEXT,
    analysed_at       TEXT,
    reply_status      TEXT NOT NULL DEFAULT 'Open',
    reply_text        TEXT,
    reply_updated_at  TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _date_text(value) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None

    return pd.Timestamp(value).strftime("%Y-%m-%d")


class AnalysisCache(MutableMapping):
    """Dict-like view of the persistent analysis cache."""

    def __init__(self, database: "Database"):
        self._db = database

    def __getitem__(self, key: str) -> dict:
        with closing(self._db.connect()) as conn:
            row = conn.execute(
                "SELECT result FROM analysis_cache WHERE key = ?", (key,)
            ).fetchone()

        if row is None:
            raise KeyError(key)

        return json.loads(row["result"])

    def __setitem__(self, key: str, value: dict) -> None:
        with closing(self._db.connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO analysis_cache (key, result, created_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(value), _now()),
            )

    def __delitem__(self, key: str) -> None:
        with closing(self._db.connect()) as conn, conn:
            deleted = conn.execute(
                "DELETE FROM analysis_cache WHERE key = ?", (key,)
            ).rowcount

        if not deleted:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        with closing(self._db.connect()) as conn:
            return conn.execute(
                "SELECT 1 FROM analysis_cache WHERE key = ?", (key,)
            ).fetchone() is not None

    def __iter__(self) -> Iterator[str]:
        with closing(self._db.connect()) as conn:
            keys = [row["key"] for row in conn.execute("SELECT key FROM analysis_cache")]

        return iter(keys)

    def __len__(self) -> int:
        with closing(self._db.connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM analysis_cache").fetchone()[0]


class Database:
    """Review history, reply workflow and analysis cache in one SQLite file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with closing(self.connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------- cache ----------

    def cache(self) -> AnalysisCache:
        return AnalysisCache(self)

    # ---------- reviews ----------

    def save_analysed(self, df: pd.DataFrame, model_id: str) -> int:
        """
        Save successfully analysed reviews (insert or update by review_id).

        Reply status and edited reply text are kept when a review is
        analysed again.
        """

        if "analysis_status" in df.columns:
            df = df[df["analysis_status"] == "success"]

        analysed_at = _now()
        rows = []

        for record in df.to_dict(orient="records"):
            rows.append(
                {
                    **{column: record.get(column) for column in REVIEW_COLUMNS},
                    "date": _date_text(record.get("date")),
                    "rating": None if pd.isna(record.get("rating")) else float(record["rating"]),
                    "model_id": model_id,
                    "analysed_at": analysed_at,
                }
            )

        if not rows:
            return 0

        columns = ", ".join(REVIEW_COLUMNS)
        placeholders = ", ".join(f":{column}" for column in REVIEW_COLUMNS)
        updates = ", ".join(
            f"{column} = excluded.{column}" for column in REVIEW_COLUMNS if column != "review_id"
        )

        with closing(self.connect()) as conn, conn:
            conn.executemany(
                f"INSERT INTO reviews ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(review_id) DO UPDATE SET {updates}",
                rows,
            )

        return len(rows)

    def load_reviews(self) -> pd.DataFrame:
        """All stored reviews, newest first."""

        with closing(self.connect()) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM reviews ORDER BY date DESC, review_id", conn
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df

    def count_reviews(self) -> int:
        with closing(self.connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]

    # ---------- replies ----------

    def load_replies(self, review_ids: list[str] | None = None) -> dict[str, dict]:
        """Reply status and text, by review_id."""

        query = "SELECT review_id, reply_status, reply_text FROM reviews"
        params: list = []

        if review_ids is not None:
            if not review_ids:
                return {}
            query += f" WHERE review_id IN ({', '.join('?' * len(review_ids))})"
            params = list(review_ids)

        with closing(self.connect()) as conn:
            return {
                row["review_id"]: {
                    "reply_status": row["reply_status"],
                    "reply_text": row["reply_text"],
                }
                for row in conn.execute(query, params)
            }

    def update_replies(self, updates: dict[str, dict]) -> None:
        """Save reply status/text for reviews that are stored."""

        rows = [
            (
                values["reply_status"] if values.get("reply_status") in REPLY_STATUSES else "Open",
                values.get("reply_text"),
                _now(),
                review_id,
            )
            for review_id, values in updates.items()
        ]

        with closing(self.connect()) as conn, conn:
            conn.executemany(
                "UPDATE reviews SET reply_status = ?, reply_text = ?, reply_updated_at = ? "
                "WHERE review_id = ?",
                rows,
            )

    # ---------- privacy ----------

    def delete_all(self) -> None:
        """Delete all stored reviews, replies and cached analyses."""

        with closing(self.connect()) as conn, conn:
            conn.execute("DELETE FROM reviews")
            conn.execute("DELETE FROM analysis_cache")

        with closing(self.connect()) as conn:
            conn.execute("VACUUM")

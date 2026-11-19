import pandas as pd
import pytest

from customer_reviews.storage import Database


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "nested" / "reviews.db")


def analysed_df(**overrides):
    row = {
        "review_id": "R1", "date": pd.Timestamp("2026-10-05"), "platform": "Google",
        "rating": 2, "review_text": "Cold food", "language": "English",
        "sentiment": "Negative", "category": "Food Quality", "aspects": "Food Quality; Delivery",
        "urgency": "MEDIUM", "summary": "Food was cold", "suggested_reply": "Sorry!",
        "analysis_status": "success", "analysis_error": "",
    }
    row.update(overrides)
    failed = dict(row, review_id="R2", analysis_status="failed", sentiment=None)
    return pd.DataFrame([row, failed])


def test_cache_behaves_like_a_dict(db):
    cache = db.cache()

    assert "k" not in cache and len(cache) == 0
    cache["k"] = {"sentiment": "Positive"}

    assert "k" in cache
    assert cache["k"] == {"sentiment": "Positive"}
    assert list(cache) == ["k"] and len(cache) == 1

    del cache["k"]
    assert "k" not in cache
    with pytest.raises(KeyError):
        cache["k"]


def test_cache_survives_reopening(tmp_path):
    Database(tmp_path / "r.db").cache()["k"] = {"a": 1}

    assert Database(tmp_path / "r.db").cache()["k"] == {"a": 1}


def test_save_and_load_only_successful_reviews(db):
    saved = db.save_analysed(analysed_df(), "openai/test")

    stored = db.load_reviews()

    assert saved == 1
    assert list(stored["review_id"]) == ["R1"]
    assert stored.loc[0, "date"] == pd.Timestamp("2026-10-05")
    assert stored.loc[0, "rating"] == 2.0
    assert stored.loc[0, "aspects"] == "Food Quality; Delivery"
    assert stored.loc[0, "model_id"] == "openai/test"
    assert stored.loc[0, "reply_status"] == "Open"


def test_reanalysis_updates_review_but_keeps_reply(db):
    db.save_analysed(analysed_df(), "m1")
    db.update_replies({"R1": {"reply_status": "Approved", "reply_text": "Edited reply"}})

    db.save_analysed(analysed_df(summary="New summary"), "m2")

    stored = db.load_reviews()
    assert stored.loc[0, "summary"] == "New summary"
    assert stored.loc[0, "model_id"] == "m2"
    assert db.load_replies(["R1"]) == {
        "R1": {"reply_status": "Approved", "reply_text": "Edited reply"}
    }


def test_invalid_reply_status_falls_back_to_open(db):
    db.save_analysed(analysed_df(), "m1")
    db.update_replies({"R1": {"reply_status": "Hacked", "reply_text": "x"}})

    assert db.load_replies()["R1"]["reply_status"] == "Open"


def test_load_replies_for_no_ids(db):
    assert db.load_replies([]) == {}


def test_delete_all(db):
    db.save_analysed(analysed_df(), "m1")
    db.cache()["k"] = {}

    db.delete_all()

    assert db.count_reviews() == 0
    assert len(db.cache()) == 0

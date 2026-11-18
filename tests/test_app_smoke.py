from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_starts_without_api_key(monkeypatch):
    """The app must load (no import-time LLM creation) even without a key."""

    monkeypatch.setattr("config.OPENAI_API_KEY", None)

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value.startswith("📊 AI-Powered Customer Review Analyzer")
    assert any("OPENAI_API_KEY" in w.value for w in at.sidebar.warning)

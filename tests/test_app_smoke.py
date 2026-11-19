from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_starts_without_api_key():
    """The app must load (no import-time LLM creation) even without a key."""

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value.startswith("📊 AI-Powered Customer Review Analyzer")
    assert any("OPENAI_API_KEY" in w.value for w in at.sidebar.warning)


def test_invalid_config_shows_error(monkeypatch):
    monkeypatch.setenv("MAX_REVIEWS", "-1")

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not at.exception
    assert "MAX_REVIEWS" in at.error[0].value


def test_password_gate(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not at.exception
    assert at.subheader[0].value == "🔒 Sign in"
    assert len(at.sidebar.warning) == 0  # nothing else rendered

    at.text_input[0].input("wrong")
    at.button[0].click().run()
    assert at.error[0].value == "Incorrect password."

    at.text_input[0].input("s3cret")
    at.button[0].click().run()
    assert at.session_state["authenticated"] is True
    assert "Sign in" not in [s.value for s in at.subheader]

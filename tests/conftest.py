import pytest
from langchain_core.runnables import RunnableLambda

from customer_reviews import config
from customer_reviews.schemas import ReviewAnalysis
from customer_reviews.services import analysis, insights


SETTINGS_ENV_VARS = [
    "LLM_PROVIDER", "MODEL_NAME", "TEMPERATURE",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
    "MAX_REVIEWS", "MAX_REVIEW_CHARS", "MAX_CONCURRENCY", "MAX_RETRIES",
]


def clear_caches():
    config.get_settings.cache_clear()
    analysis.get_review_analysis_chain.cache_clear()
    insights.get_weekly_insights_chain.cache_clear()


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Tests never read the developer's .env file or real API keys."""

    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(config, "_streamlit_secrets", lambda: {})

    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    clear_caches()
    yield
    clear_caches()


def make_analysis(**overrides) -> ReviewAnalysis:
    values = {
        "sentiment": "Negative",
        "category": "Delivery",
        "urgency": "MEDIUM",
        "summary": "Late delivery.",
        "suggested_reply": "Sorry for the delay.",
    }
    values.update(overrides)
    return ReviewAnalysis(**values)


@pytest.fixture
def fake_chain():
    """A chain that fails for review_text containing 'FAIL', else succeeds."""

    calls = []

    def run(inputs: dict) -> ReviewAnalysis:
        calls.append(inputs)
        if "FAIL" in inputs["review_text"]:
            raise RuntimeError("simulated rate limit")
        return make_analysis(summary=f"Summary of {inputs['review_id']}")

    chain = RunnableLambda(run)
    chain.calls = calls
    return chain

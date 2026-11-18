import pytest
from langchain_core.runnables import RunnableLambda

from schemas import ReviewAnalysis


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

"""Ask tab: chat with your reviews."""

import pandas as pd
import streamlit as st

from customer_reviews.config import Settings
from customer_reviews.core.safety import sanitize_markdown
from customer_reviews.services.chat import answer_question
from customer_reviews.ui.common import friendly_error


EXAMPLE_QUESTIONS = [
    "What are the top 3 complaints?",
    "Which platform has the most negative reviews?",
    "What do customers like most?",
    "Summarise the delivery issues.",
]

_ROLE_FOR_MODEL = {"user": "human", "assistant": "ai"}


def render_ask_tab(df: pd.DataFrame, settings: Settings) -> None:
    history: list[tuple[str, str]] = st.session_state.chat_history or []

    st.caption(
        f"Answers are based on the {len(df)} reviews matching the sidebar filters "
        f"(up to {settings.chat_max_reviews} most recent). AI answers can be wrong: "
        "check the cited review IDs."
    )

    if not history:
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, example in zip(cols, EXAMPLE_QUESTIONS, strict=True):
            if col.button(example, disabled=not settings.has_api_key(), width="stretch"):
                st.session_state.pending_question = example

    for role, content in history:
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input(
        "Ask a question about your reviews…",
        disabled=not settings.has_api_key() or df.empty,
    )
    question = question or st.session_state.pop("pending_question", None)

    if history and st.button("Clear conversation", icon=":material/delete:"):
        st.session_state.chat_history = None
        st.rerun()

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                answer = answer_question(
                    question,
                    df,
                    [(_ROLE_FOR_MODEL[role], content) for role, content in history],
                    max_reviews=settings.chat_max_reviews,
                )
        except Exception as e:
            st.error(friendly_error("Answering", e))
            return

    st.session_state.chat_history = [
        *history,
        ("user", question),
        ("assistant", sanitize_markdown(answer)),
    ]
    st.rerun()

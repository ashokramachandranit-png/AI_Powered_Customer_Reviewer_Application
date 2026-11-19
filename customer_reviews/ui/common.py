"""Session state, theme detection and user-facing messages."""

import streamlit as st

from customer_reviews.config import ConfigError, Settings
from customer_reviews.storage import Database


STATE_DEFAULTS = {
    "upload_signature": None,
    "upload_error": None,
    "validated_reviews": None,
    "rejected_reviews": None,
    "analysed_reviews": None,
    "weekly_report": None,
    "flash": None,
    "chat_history": None,
}

# Prefix of widget keys that must be reset when a new file is uploaded
FILTER_KEY_PREFIX = "filter_"


def init_state() -> None:
    """Create session state keys on first run."""

    for key, value in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    # Analysis results by content hash, used when persistence is off. Kept
    # across uploads so re-uploading the same reviews does not call the LLM.
    st.session_state.setdefault("analysis_cache", {})

    # Reply workflow: {review_id: {"reply_status": ..., "reply_text": ...}}
    st.session_state.setdefault("replies", {})


@st.cache_resource(show_spinner=False)
def _open_database(path: str) -> Database:
    return Database(path)


def get_database(settings: Settings) -> Database | None:
    """The shared SQLite database, or None when persistence is off or fails."""

    if settings.database_file is None:
        return None

    try:
        return _open_database(str(settings.database_file))
    except Exception as error:
        st.sidebar.warning(
            f"Saved history is unavailable ({type(error).__name__}). "
            "Results will only be kept for this session."
        )
        return None


def analysis_cache(db: Database | None):
    """Persistent cache when the database is available, else per-session."""

    return db.cache() if db is not None else st.session_state.analysis_cache


def remember_results(analysed, settings: Settings, db: Database | None) -> None:
    """Save analysed reviews and load their saved reply workflow state."""

    if db is None:
        return

    db.save_analysed(analysed, settings.model_id)
    st.session_state.replies.update(db.load_replies(list(analysed["review_id"])))


def reset_for_new_file() -> None:
    """Clear results and filters that belong to a previously uploaded file."""

    for key, value in STATE_DEFAULTS.items():
        if key != "upload_signature":
            st.session_state[key] = value

    for key in list(st.session_state.keys()):
        if str(key).startswith(FILTER_KEY_PREFIX):
            del st.session_state[key]


def current_theme() -> str:
    """'dark' or 'light', following the viewer's Streamlit theme."""

    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def friendly_error(action: str, error: Exception) -> str:
    """Turn an exception into a short message for the user."""

    # Our own configuration and validation messages are safe and helpful
    if isinstance(error, ConfigError) or type(error) is ValueError:
        return str(error)

    return (
        f"{action} failed ({type(error).__name__}). "
        "Please try again. If the problem continues, check your API key, "
        "model name and internet connection."
    )


def set_flash(kind: str, message: str) -> None:
    """Show a message after the next rerun."""

    st.session_state.flash = (kind, message)


def show_flash() -> None:
    flash = st.session_state.get("flash")

    if flash:
        kind, message = flash
        getattr(st, kind)(message)
        st.session_state.flash = None

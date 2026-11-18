"""Session state, theme detection and user-facing messages."""

import streamlit as st

from customer_reviews.config import ConfigError


STATE_DEFAULTS = {
    "upload_signature": None,
    "upload_error": None,
    "validated_reviews": None,
    "rejected_reviews": None,
    "analysed_reviews": None,
    "weekly_report": None,
    "flash": None,
}

# Prefix of widget keys that must be reset when a new file is uploaded
FILTER_KEY_PREFIX = "filter_"


def init_state() -> None:
    """Create session state keys on first run."""

    for key, value in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    # Analysis results by content hash. Kept across uploads so re-uploading
    # the same reviews does not call the LLM again.
    st.session_state.setdefault("analysis_cache", {})


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

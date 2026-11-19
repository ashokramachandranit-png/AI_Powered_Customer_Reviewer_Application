"""Optional password protection (APP_PASSWORD)."""

import hmac
import time

import streamlit as st

from customer_reviews.config import Settings


def require_password(settings: Settings) -> None:
    """Stop the app until the viewer enters APP_PASSWORD (if one is set)."""

    password = settings.password

    if password is None or st.session_state.get("authenticated"):
        return

    st.subheader("🔒 Sign in")

    with st.form("login", border=True):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        if hmac.compare_digest(entered.encode("utf-8"), password.encode("utf-8")):
            st.session_state.authenticated = True
            st.rerun()

        # Slow down password guessing
        time.sleep(1)
        st.error("Incorrect password.")

    st.stop()


def render_logout(settings: Settings) -> None:
    if settings.password is None:
        return

    with st.sidebar:
        if st.button("Sign out", icon=":material/logout:"):
            st.session_state.clear()
            st.rerun()

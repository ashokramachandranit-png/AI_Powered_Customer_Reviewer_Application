"""
AI-Powered Customer Review Analyzer - Streamlit entry point.

Run with:  streamlit run app.py
"""

import streamlit as st

from customer_reviews.config import ConfigError, get_settings
from customer_reviews.services.filters import apply_filters, successful_reviews
from customer_reviews.ui.analysis_tab import render_analysis_tab
from customer_reviews.ui.ask_tab import render_ask_tab
from customer_reviews.ui.auth import render_logout, require_password
from customer_reviews.ui.common import current_theme, get_database, init_state
from customer_reviews.ui.dashboard import render_dashboard_tab
from customer_reviews.ui.data_tab import handle_upload, render_data_tab
from customer_reviews.ui.history_tab import render_history_tab
from customer_reviews.ui.insights_tab import render_insights_tab
from customer_reviews.ui.replies_tab import render_replies_tab
from customer_reviews.ui.reviews_tab import render_explore_tab, render_urgent_tab
from customer_reviews.ui.sidebar import render_filters, render_upload


st.set_page_config(
    page_title="AI-Powered Customer Review Analyzer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI-Powered Customer Review Analyzer")

try:
    settings = get_settings()
except ConfigError as error:
    st.error(str(error))
    st.stop()

require_password(settings)

st.write(
    "Analyse customer reviews using AI, identify customer sentiment, detect "
    "urgent issues, and generate business insights."
)

init_state()
db = get_database(settings)
theme = current_theme()

uploaded_file = render_upload(settings)
render_logout(settings)
handle_upload(uploaded_file)

if uploaded_file is None:
    st.info(
        "**Get started:** upload a customer review CSV from the sidebar. "
        "Not sure about the format? Download the sample CSV there.\n\n"
        "1. **Data** – check which rows are valid\n"
        "2. **Analysis** – let AI classify sentiment, topics, language and urgency\n"
        "3. **Dashboard, Insights, Urgent, Replies, Ask, Explore** – act on the results",
        icon=":material/upload_file:",
    )

    if db is not None and db.count_reviews():
        st.subheader("Saved history")
        render_history_tab(db, theme)

    st.stop()


analysed = st.session_state.analysed_reviews
succeeded = successful_reviews(analysed) if analysed is not None else None

filters = render_filters(succeeded) if succeeded is not None and not succeeded.empty else None

tab_names = ["Data", "Analysis", "Dashboard", "Insights", "Urgent", "Replies", "Ask", "Explore"]
if db is not None:
    tab_names.append("History")

tabs = dict(zip(tab_names, st.tabs(tab_names, key="main_tabs"), strict=True))

with tabs["Data"]:
    render_data_tab()

with tabs["Analysis"]:
    render_analysis_tab(settings, db)

if db is not None:
    with tabs["History"]:
        render_history_tab(db, theme)

result_tabs = ["Dashboard", "Insights", "Urgent", "Replies", "Ask", "Explore"]

if succeeded is None or succeeded.empty:
    message = (
        "Run the analysis first (Analysis tab)." if analysed is None
        else "No reviews were analysed successfully. Retry them in the Analysis tab."
    )
    for name in result_tabs:
        with tabs[name]:
            st.info(message)
    st.stop()

filtered = apply_filters(succeeded, filters)

with tabs["Dashboard"]:
    render_dashboard_tab(filtered, len(succeeded), filters.is_active(), theme)

with tabs["Insights"]:
    render_insights_tab(succeeded, can_call_llm=settings.has_api_key())

with tabs["Urgent"]:
    render_urgent_tab(succeeded, settings)

with tabs["Replies"]:
    render_replies_tab(succeeded, db)

with tabs["Ask"]:
    render_ask_tab(filtered, settings)

with tabs["Explore"]:
    render_explore_tab(filtered, len(succeeded))

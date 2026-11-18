# AI-Powered Customer Review Analyzer

A Streamlit web app that uses an LLM (OpenAI by default, or Anthropic Claude
or Google Gemini, via LangChain) to analyse customer reviews. Upload a CSV of
reviews and the app will:

- **Validate** the file and show which rows were skipped and why
- **Classify** every review: sentiment, category, urgency (HIGH / MEDIUM / LOW),
  a short English summary and a suggested reply, with a live progress bar and
  a token estimate before you start
- Show an **analytics dashboard**: sentiment, category, urgency and platform
  charts plus weekly sentiment and rating trends (light and dark themes)
- Generate a **business insights report** in Markdown
- **Filter** reviews by platform, sentiment, category, urgency, date range and
  text search, and list **high-urgency complaints** for human follow-up
- **Download** all results, filtered results, urgent complaints and the report

The app is organised in tabs: **Data → Analysis → Dashboard → Insights →
Urgent → Explore**. Results are cached per review, so re-analysing the same
reviews with the same model does not call the LLM again.

## Quick start

Requires Python 3.11 or newer.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env             # Windows: copy .env.example .env
#    then edit .env and set OPENAI_API_KEY (or another provider, see below)

# 4. Run the app
streamlit run app.py
```

Open http://localhost:8501, then upload `sample_data/sample_reviews.csv`
(or download it from the app's sidebar) to try it out.

## CSV format

| Column        | Description                                  | Example        |
|---------------|----------------------------------------------|----------------|
| `review_id`   | Unique ID for the review                     | `R001`         |
| `date`        | Review date (most common formats work)       | `2026-10-05`   |
| `platform`    | Where the review was posted                  | `Google`       |
| `rating`      | Star rating from 1 to 5                      | `4`            |
| `review_text` | The customer's review                        | `Great food!`  |

Rows with missing values, invalid dates, ratings outside 1-5 or duplicate
`review_id`s are skipped. The app lists every skipped row with the reason
and lets you download them.

## Configuration

Settings are read from environment variables, then the `.env` file (see
`.env.example`), then Streamlit secrets (`.streamlit/secrets.toml` or the
Streamlit Cloud *Secrets* panel). Invalid values are reported in the app.

| Variable            | Default        | Purpose                                        |
|---------------------|----------------|------------------------------------------------|
| `LLM_PROVIDER`      | `openai`       | `openai`, `anthropic` or `gemini`              |
| `OPENAI_API_KEY`    | –              | Needed when `LLM_PROVIDER=openai`              |
| `ANTHROPIC_API_KEY` | –              | Needed when `LLM_PROVIDER=anthropic`           |
| `GOOGLE_API_KEY`    | –              | Needed when `LLM_PROVIDER=gemini`              |
| `MODEL_NAME`        | per provider   | `gpt-4.1-mini`, `claude-opus-5-5`, `gemini-2.5-flash` |
| `TEMPERATURE`       | `0.2`          | Lower = more consistent (not used for Claude)  |
| `MAX_REVIEWS`       | `500`          | Maximum reviews analysed per run (cost limit)  |
| `MAX_REVIEW_CHARS`  | `4000`         | Longer reviews are truncated before sending    |
| `MAX_CONCURRENCY`   | `5`            | Parallel requests to the LLM                   |
| `MAX_RETRIES`       | `3`            | Attempts per review before it is marked failed |

The maximum upload size (10 MB) is set in `.streamlit/config.toml`.

Model names change over time: check your provider's documentation and set
`MODEL_NAME` if a default is no longer available.

## Project structure

```
app.py                          Thin Streamlit entry point (page setup, tabs)
customer_reviews/
  config.py                     Typed settings (pydantic-settings)
  llm.py                        LLM provider factory (OpenAI / Claude / Gemini)
  schemas.py                    Pydantic model for the AI's structured output
  prompts/                      LangChain prompt templates (+ PROMPT_VERSION)
  core/
    validation.py               CSV validation with a report of skipped rows
    safety.py                   CSV formula-injection and Markdown link protection
  services/
    analysis.py                 Prompt inputs, LLM chain, parallel analysis,
                                caching, progress and usage estimates
    insights.py                 Aggregated statistics and the insights report
    filters.py                  Review filtering
  ui/
    common.py                   Session state, theme detection, messages
    sidebar.py                  Upload controls and filters
    charts.py                   Plotly chart builders (theme-aware colours)
    data_tab.py, analysis_tab.py, dashboard.py,
    insights_tab.py, reviews_tab.py   One module per tab
sample_data/                    Example CSV
tests/                          Pytest test suite (no API key needed)
```

The `services` and `core` layers have no Streamlit code, so they can be
reused (e.g. from a script or an API) and tested on their own.

## How it works

1. `core.validation` cleans the uploaded CSV and reports rejected rows.
2. `services.analysis` sends each review through the analysis chain in
   parallel. Failed reviews (e.g. rate limits) are retried automatically, and
   any that still fail are marked as failed instead of stopping the whole run.
   You can retry them from the Analysis tab.
3. Results are merged back into the reviews by `review_id` and cached by a
   hash of the review, model and prompt version.
4. The dashboard, insights and urgent list use only successfully analysed
   reviews.

## Security and privacy

- **Review text is sent to your LLM provider.** Remove personal data (names, phone
  numbers, emails) from reviews if your policies require it.
- Reviews are treated as untrusted input: they are wrapped in delimiters and
  the prompt instructs the model to ignore instructions inside them.
- Downloaded CSVs are protected against spreadsheet formula injection.
- Links and images are removed from the AI-generated report.
- Suggested replies are AI-generated. Always review them before sending.
- Never commit your `.env` file (it is listed in `.gitignore`).

## Development

```bash
pip install -r requirements-dev.txt
ruff check .     # lint
pytest           # run tests (uses fake LLMs, no API key needed)
```

GitHub Actions runs lint and tests on every push and pull request.

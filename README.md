# AI-Powered Customer Review Analyzer

A Streamlit web app that uses an LLM (OpenAI via LangChain) to analyse
customer reviews. Upload a CSV of reviews and the app will:

- **Validate** the file and show which rows were skipped and why
- **Classify** every review: sentiment, category, urgency (HIGH / MEDIUM / LOW),
  a short English summary and a suggested reply
- Show an **analytics dashboard** (sentiment, category, urgency and platform charts)
- Generate a **business insights report** in Markdown
- **Filter** reviews and list **high-urgency complaints** for human follow-up
- **Download** the analysed reviews (CSV) and the report (Markdown)

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
#    then edit .env and set OPENAI_API_KEY

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

All settings live in `.env` (see `.env.example`):

| Variable           | Default        | Purpose                                       |
|--------------------|----------------|-----------------------------------------------|
| `OPENAI_API_KEY`   | *(required)*   | Your OpenAI API key                           |
| `MODEL_NAME`       | `gpt-4.1-mini` | OpenAI model used for analysis and reports    |
| `TEMPERATURE`      | `0.2`          | Lower = more consistent output                |
| `MAX_REVIEWS`      | `500`          | Maximum reviews analysed per run (cost limit) |
| `MAX_REVIEW_CHARS` | `4000`         | Longer reviews are truncated before sending   |
| `MAX_CONCURRENCY`  | `5`            | Parallel requests to the LLM                  |
| `MAX_RETRIES`      | `3`            | Attempts per review before it is marked failed |

The maximum upload size (10 MB) is set in `.streamlit/config.toml`.

## Project structure

```
app.py                      Streamlit UI and page flow
config.py                   Settings from .env and LLM creation
schemas.py                  Pydantic model for the AI's structured output
data_validator.py           CSV validation with a report of skipped rows
review_analyzer.py          Review analysis chain (prompt -> LLM -> schema)
batch_processor.py          Parallel analysis with per-review error handling
insights_generator.py       Aggregated statistics and the insights report
dashboard.py                Summary metrics and Plotly charts
output_safety.py            CSV formula-injection and Markdown link protection
prompts/                    LangChain prompt templates
sample_data/                Example CSV
tests/                      Pytest test suite (no API key needed)
```

## How it works

1. `data_validator` cleans the uploaded CSV and reports rejected rows.
2. `batch_processor` sends each review through the analysis chain in parallel.
   Failed reviews (e.g. rate limits) are retried automatically, and any that
   still fail are marked as failed instead of stopping the whole run. You can
   retry them from the UI.
3. Results are merged back into the reviews by `review_id`.
4. `dashboard` and `insights_generator` use only successfully analysed reviews.

## Security and privacy

- **Review text is sent to OpenAI.** Remove personal data (names, phone
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

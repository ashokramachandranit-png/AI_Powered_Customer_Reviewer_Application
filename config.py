import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()


def _int_env(name: str, default: int) -> int:
    """Read a positive integer setting, falling back to the default."""

    try:
        value = int(os.getenv(name, default))
    except ValueError:
        return default

    return value if value > 0 else default


# Read configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Safety and cost limits
MAX_REVIEWS = _int_env("MAX_REVIEWS", 500)            # reviews analysed per run
MAX_REVIEW_CHARS = _int_env("MAX_REVIEW_CHARS", 4000)  # characters sent per review
MAX_CONCURRENCY = _int_env("MAX_CONCURRENCY", 5)       # parallel LLM requests
MAX_RETRIES = _int_env("MAX_RETRIES", 3)               # attempts per review


class MissingAPIKeyError(ValueError):
    """Raised when the LLM API key has not been configured."""


def has_api_key() -> bool:
    """Return True when an OpenAI API key is configured."""

    return bool(OPENAI_API_KEY)


def get_llm():
    """Create and return the OpenAI chat model."""

    if not OPENAI_API_KEY:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env "
            "and add your OpenAI API key, then restart the app."
        )

    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY
    )

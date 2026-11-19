"""
LLM provider factory.

Supports OpenAI (default), Anthropic Claude and Google Gemini via LangChain.
Provider packages are imported only when selected, so the optional ones do
not need to be installed unless used.
"""

from pydantic import BaseModel

from customer_reviews.config import ConfigError, Settings


# Max output tokens for Claude. Current Claude models think before
# answering, and thinking counts towards this limit.
ANTHROPIC_MAX_TOKENS = 16000


def _missing_package(package: str, provider: str) -> ConfigError:
    return ConfigError(
        f"LLM_PROVIDER is '{provider}' but the '{package}' package is not "
        f"installed. Run: pip install {package}"
    )


def create_chat_model(settings: Settings):
    """Create the LangChain chat model for the configured provider."""

    api_key = settings.api_key()
    provider = settings.llm_provider

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise _missing_package("langchain-openai", provider) from error

        return ChatOpenAI(
            model=settings.model,
            temperature=settings.temperature,
            api_key=api_key,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as error:
            raise _missing_package("langchain-anthropic", provider) from error

        # TEMPERATURE is intentionally not passed: current Claude models
        # reject sampling parameters (temperature/top_p/top_k).
        return ChatAnthropic(
            model=settings.model,
            api_key=api_key,
            max_tokens=ANTHROPIC_MAX_TOKENS,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as error:
            raise _missing_package("langchain-google-genai", provider) from error

        return ChatGoogleGenerativeAI(
            model=settings.model,
            temperature=settings.temperature,
            google_api_key=api_key,
        )

    raise ConfigError(f"Unsupported LLM_PROVIDER: {provider}")


def with_structured_output(llm, schema: type[BaseModel], settings: Settings):
    """
    Make the model return `schema` objects.

    Claude uses native JSON-schema structured outputs: LangChain's default
    for Anthropic forces a tool call, which current Claude models reject.
    """

    if settings.llm_provider == "anthropic":
        return llm.with_structured_output(schema, method="json_schema")

    return llm.with_structured_output(schema)

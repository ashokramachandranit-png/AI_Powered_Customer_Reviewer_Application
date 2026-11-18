"""
Application settings.

Values are read, in order of priority, from:
1. environment variables
2. the project's .env file
3. Streamlit secrets (.streamlit/secrets.toml or the Streamlit Cloud
   secrets panel), for values not set above
4. the defaults below
"""

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

Provider = Literal["openai", "anthropic", "gemini"]

# Default model per provider (override with MODEL_NAME)
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-opus-5-5",
    "gemini": "gemini-2.5-flash",
}

# Environment variable holding the API key for each provider
API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


class ConfigError(ValueError):
    """Raised when the configuration is invalid."""


class MissingAPIKeyError(ConfigError):
    """Raised when the API key for the selected provider is not configured."""


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    llm_provider: Provider = "openai"
    model_name: str = ""
    temperature: float | None = Field(default=0.2, ge=0, le=2)

    # API keys (only the selected provider's key is needed)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # Safety and cost limits
    max_reviews: int = Field(default=500, gt=0)
    max_review_chars: int = Field(default=4000, gt=0)
    max_concurrency: int = Field(default=5, gt=0, le=50)
    max_retries: int = Field(default=3, gt=0, le=10)

    @model_validator(mode="before")
    @classmethod
    def _empty_strings_mean_default(cls, values):
        """Treat `NAME=` (empty) in .env as "not set"."""

        if isinstance(values, dict):
            return {
                key: value for key, value in values.items()
                if not (isinstance(value, str) and value.strip() == "")
            }
        return values

    @property
    def model(self) -> str:
        """The model to use: MODEL_NAME or the provider's default."""

        return self.model_name or DEFAULT_MODELS[self.llm_provider]

    @property
    def model_id(self) -> str:
        """Provider and model, e.g. 'openai/gpt-4.1-mini'."""

        return f"{self.llm_provider}/{self.model}"

    @property
    def api_key_env_var(self) -> str:
        return API_KEY_ENV_VARS[self.llm_provider]

    def _api_key_secret(self) -> SecretStr | None:
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.google_api_key,
        }[self.llm_provider]

    def has_api_key(self) -> bool:
        secret = self._api_key_secret()
        return bool(secret and secret.get_secret_value().strip())

    def api_key(self) -> str:
        """Return the selected provider's API key or raise MissingAPIKeyError."""

        if not self.has_api_key():
            raise MissingAPIKeyError(
                f"{self.api_key_env_var} is missing. Copy .env.example to .env, "
                f"add your API key for '{self.llm_provider}', then restart the app."
            )

        return self._api_key_secret().get_secret_value().strip()


def _streamlit_secrets() -> dict:
    """Read top-level Streamlit secrets, if any are configured."""

    try:
        import streamlit as st

        return {
            key.lower(): value
            for key, value in st.secrets.items()
            if not isinstance(value, Mapping)
        }
    except Exception:
        # No secrets file, or not running inside Streamlit
        return {}


def load_settings(env_file: Path | None = None, use_secrets: bool = True) -> Settings:
    """Build Settings, turning validation errors into a readable ConfigError."""

    if env_file is None:
        env_file = ENV_FILE

    overrides = {}

    if use_secrets:
        # Secrets fill in values that are not set as environment variables
        overrides = {
            key: value for key, value in _streamlit_secrets().items()
            if key.upper() not in os.environ and key in Settings.model_fields
        }

    env_file = env_file if env_file and Path(env_file).exists() else None

    try:
        return Settings(_env_file=env_file, **overrides)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']).upper()}: {e['msg']}"
            for e in error.errors()
        )
        raise ConfigError(f"Invalid configuration - {problems}") from error


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings (loaded once per process)."""

    return load_settings()

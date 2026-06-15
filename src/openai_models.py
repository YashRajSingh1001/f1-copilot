"""OpenAI model defaults and small helpers used across the app."""

from openai import OpenAI

from .config import get


DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_TEXT_VERBOSITY = "low"


def openai_model() -> str:
    return get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def openai_api_key() -> str:
    return get("OPENAI_API_KEY")


def embedding_model() -> str:
    return get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def reasoning_effort() -> str:
    return get("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)


def text_verbosity() -> str:
    return get("OPENAI_TEXT_VERBOSITY", DEFAULT_TEXT_VERBOSITY)


def max_output_tokens(default: int) -> int:
    raw_value = get("OPENAI_MAX_OUTPUT_TOKENS", str(default))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def openai_client() -> OpenAI:
    return OpenAI(api_key=openai_api_key())

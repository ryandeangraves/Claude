"""Configuration for the OpenAI agent.

The API key is read from the ``OPENAI_API_KEY`` environment variable and is
never logged, printed, or included in error messages.
"""
import os
from dataclasses import dataclass
from typing import Mapping, Optional

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_AUDITOR_MODEL = "gpt-6-astra"
DEFAULT_IMAGE_MODEL = "gpt-image-1"
DEFAULT_MAX_TURNS = 10


class ConfigError(Exception):
    """Raised when the agent cannot be configured from the environment."""


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    auditor_model: str = DEFAULT_AUDITOR_MODEL
    image_model: str = DEFAULT_IMAGE_MODEL
    max_turns: int = DEFAULT_MAX_TURNS
    tracing_enabled: bool = True

    def __repr__(self) -> str:  # never leak the key in logs / tracebacks
        return (
            f"AgentConfig(api_key='***', model={self.model!r}, "
            f"auditor_model={self.auditor_model!r}, image_model={self.image_model!r}, "
            f"max_turns={self.max_turns}, tracing_enabled={self.tracing_enabled})"
        )


def _looks_like_openai_key(value: str) -> bool:
    return value.startswith("sk-") and len(value) > 20


def load_config(env: Optional[Mapping[str, str]] = None) -> AgentConfig:
    """Build an :class:`AgentConfig` from environment variables.

    Recognised variables:

    * ``OPENAI_API_KEY``      (required)
    * ``OPENAI_AGENT_MODEL``  (optional, default ``gpt-4.1-mini``) - assistant + image orchestration
    * ``OPENAI_AUDITOR_MODEL`` (optional, default ``gpt-6-astra``) - auditor (needs vision)
    * ``OPENAI_IMAGE_MODEL``  (optional, default ``gpt-image-1``) - image_generation tool
    * ``OPENAI_AGENT_MAX_TURNS`` (optional, default 10)
    * ``OPENAI_AGENT_TRACING`` (optional, ``0``/``false`` disables tracing)
    """
    env = os.environ if env is None else env

    api_key = (env.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Export it in your shell or add it to a "
            ".env file (see .env.example)."
        )
    if not _looks_like_openai_key(api_key):
        raise ConfigError(
            "OPENAI_API_KEY does not look like an OpenAI key (expected it to "
            "start with 'sk-')."
        )

    model = (env.get("OPENAI_AGENT_MODEL") or DEFAULT_MODEL).strip()
    auditor_model = (env.get("OPENAI_AUDITOR_MODEL") or DEFAULT_AUDITOR_MODEL).strip()
    image_model = (env.get("OPENAI_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL).strip()

    raw_turns = (env.get("OPENAI_AGENT_MAX_TURNS") or "").strip()
    if raw_turns:
        try:
            max_turns = int(raw_turns)
        except ValueError as exc:
            raise ConfigError("OPENAI_AGENT_MAX_TURNS must be an integer") from exc
        if max_turns < 1:
            raise ConfigError("OPENAI_AGENT_MAX_TURNS must be at least 1")
    else:
        max_turns = DEFAULT_MAX_TURNS

    tracing_flag = (env.get("OPENAI_AGENT_TRACING") or "1").strip().lower()
    tracing_enabled = tracing_flag not in {"0", "false", "no", "off"}

    return AgentConfig(
        api_key=api_key,
        model=model,
        auditor_model=auditor_model,
        image_model=image_model,
        max_turns=max_turns,
        tracing_enabled=tracing_enabled,
    )

"""Agent settings: models, effort, budget and the Anthropic key (env or .env, never code)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

INVESTIGATOR_MODEL = "claude-opus-5-5"
CRITIC_MODEL = "claude-sonnet-5-5"


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    investigator_model: str = Field(
        default=INVESTIGATOR_MODEL, validation_alias="SECOPS_INVESTIGATOR_MODEL"
    )
    critic_model: str = Field(default=CRITIC_MODEL, validation_alias="SECOPS_CRITIC_MODEL")
    effort: Literal["low", "medium", "high", "max"] = Field(
        default="medium", validation_alias="SECOPS_AGENT_EFFORT"
    )
    tool_budget: int = Field(default=12, ge=1, le=50, validation_alias="SECOPS_TOOL_BUDGET")


@lru_cache(maxsize=1)
def get_agent_settings() -> AgentSettings:
    return AgentSettings()

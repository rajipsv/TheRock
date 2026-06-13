from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.env_loader import load_agent_env

_AGENT_ROOT = Path(__file__).resolve().parents[1]
_THEROCK_ROOT = _AGENT_ROOT.parent.parent

load_agent_env()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(_AGENT_ROOT / ".env"),
            str(_THEROCK_ROOT / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "BASE_URL"),
    )
    llm_api_key: str = Field(
        default="abc-123",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_model: str = Field(
        default="Qwen3-30B-A3B",
        validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL"),
    )
    use_llm: bool = True
    llm_timeout_seconds: float = 180.0
    use_llm_alignment_fallback: bool = True
    alignment_score_threshold: float = 0.4

    github_repo: str = "kornosk/GDPR-similarity-comparison"
    github_branch: str = "main"
    github_cache_dir: str = "data/github/gdpr-similarity-comparison"


settings = Settings()

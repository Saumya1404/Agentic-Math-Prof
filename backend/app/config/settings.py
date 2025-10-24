# backend/app/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

# Resolve .env relative to repo root (project root is 3 parents up from this file)
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
if not ENV_FILE.exists():
    # Fallback to CWD .env for runs where project root isn't 3 levels up
    ENV_FILE = Path(".env").resolve()


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    GROQ_API_KEY: str
    FIRECRAWL_API_KEY: str  # Added for WebSearch MCP server
    TAVILY_API_KEY: str  # Added for potential future use
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore unexpected env vars (e.g., tavily_api_key)
    )


@lru_cache()
def get_settings() -> Settings:
    """Cache settings so they're only loaded once."""
    return Settings()


settings = get_settings()
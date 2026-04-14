# app/config.py
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    llm_model: str = "claude-opus-4-6"
    app_env: str = "development"
    block_threshold: float = 0.6  # prompts at or above this score are blocked

    model_config = {
        "env_file": Path(__file__).resolve().parent.parent / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    llm_model: str = "claude-opus-4-6"
    app_env: str = "development"
    block_threshold: float = 0.6   # prompts at or above this score are blocked
    regex_weight: float = 0.6      # weight for rule-based score in blended result
    semantic_weight: float = 0.4   # weight for semantic score in blended result

    model_config = {
        "env_file": Path(__file__).resolve().parent.parent / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

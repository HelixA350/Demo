"""
Конфигурация приложения через Pydantic BaseSettings.
Читает переменные из файла .env.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения, считываемые из переменных окружения."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_whisper_model: str = "whisper-1"

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://redis:6379/0"
    # TTL сессионного кэша (пара user_id → api_key_hash) в секундах
    auth_cache_ttl: int = 900  # 15 минут

    # Security
    secret_key: str

    # App
    vectorstore_path: str = "./vectorstore"
    document_path: str = "./data/document.pdf"
    memory_window_size: int = 10


@lru_cache
def get_settings() -> Settings:
    """Возвращает синглтон настроек приложения."""
    return Settings()

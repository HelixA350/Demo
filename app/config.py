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
    openai_api_key: str                                     # Ключ для API OpenAI
    openai_base_url: str = "https://litellm.tokengate.ru/v1"# Базовый URL API, для использования разных моделей через совместимость с OpenAI API
    openai_chat_model: str = "gpt-4o"                       # Модель LLM
    openai_embedding_model: str = "text-embedding-3-small"  # Модель векторизации
    openai_whisper_model: str = "whisper-1"                 # Модель транскрибации
    openai_max_tokens: int = 1024                           # Максимум токенов в ответе ИИ

    # Database
    database_url: str                                       # URL для подключения к базе данных
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "consultant"

    # Redis
    redis_url: str = "redis://redis:6379/0"                 # URL для подключения к Redis
    auth_cache_ttl: int = 900  # 15 минут                   # Время хранения кэша авторизации

    # Security
    secret_key: str

    # App
    vectorstore_path: str = "./vectorstore"                 # Путь к векторному хранилищу
    document_path: str = "./data/document.pdf"              # Путь к документу для базы знаний
    memory_window_size: int = 5                             # Хранимое количество пар сообщений в истории диалога


@lru_cache
def get_settings() -> Settings:
    """Возвращает синглтон настроек приложения."""
    return Settings()

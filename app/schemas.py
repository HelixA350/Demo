"""
Pydantic v2 схемы для запросов и ответов API.
"""

from pydantic import BaseModel


class RegisterResponse(BaseModel):
    """Ответ на регистрацию пользователя."""

    user_id: str
    api_key: str
    message: str


class ChatTextResponse(BaseModel):
    """Ответ на текстовый/визуальный запрос."""

    answer: str
    source_chunks: list[str]


class ChatAudioResponse(BaseModel):
    """Ответ на аудио-запрос."""

    transcription: str
    answer: str
    source_chunks: list[str]


class ClearMemoryResponse(BaseModel):
    """Ответ на очистку памяти."""

    message: str

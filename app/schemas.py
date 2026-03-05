"""
Pydantic v2 схемы для запросов и ответов API.
"""

from pydantic import BaseModel


class RegisterResponse(BaseModel):
    """Ответ на регистрацию пользователя."""

    user_id: str
    api_key: str
    message: str


class SourceChunk(BaseModel):
    """Один найденный чанк из векторной базы с указанием источника."""

    source: str
    content: str
    confidence_score: float


class ChatTextResponse(BaseModel):
    """Ответ на текстовый/визуальный запрос."""

    content: str
    used_chunk_indices: list[int]
    source_chunks: list[SourceChunk]


class ChatAudioResponse(BaseModel):
    """Ответ на аудио-запрос."""

    transcription: str
    content: str
    used_chunk_indices: list[int]
    source_chunks: list[SourceChunk]


class ClearMemoryResponse(BaseModel):
    """Ответ на очистку памяти."""

    message: str

"""
Сервис транскрипции аудио через OpenAI Whisper.
"""

import logging

import openai

from app.config import get_settings
from app.exceptions import TranscriptionError

logger = logging.getLogger(__name__)

ALLOWED_AUDIO_EXTENSIONS = {".mp3"}
ALLOWED_AUDIO_CONTENT_TYPES = {"audio/mpeg", "audio/mp3"}
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 МБ — лимит Whisper


async def transcribe(audio_bytes: bytes, filename: str) -> str:
    """
    Транскрибирует аудио-файл (mp3) в текст через OpenAI Whisper.

    Args:
        audio_bytes: содержимое аудио-файла в байтах.
        filename: имя файла (используется для определения формата).

    Returns:
        Строка с транскрибированным текстом.

    Raises:
        TranscriptionError: при ошибке API OpenAI.
        ValueError: если формат файла не поддерживается или файл слишком большой.
    """
    settings = get_settings()

    # Проверяем размер
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise ValueError(
            f"Аудио-файл слишком большой: {len(audio_bytes)} байт. "
            f"Максимум: {MAX_AUDIO_SIZE_BYTES} байт (25 МБ)."
        )

    logger.info("Транскрипция аудио: %s (%d байт)", filename, len(audio_bytes))

    try:
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        response = await client.audio.transcriptions.create(
            model=settings.openai_whisper_model,
            file=(filename, audio_bytes, "audio/mpeg"),
        )
        text = response.text
        logger.info("Транскрипция завершена: %d символов", len(text))
        return text
    except openai.OpenAIError as exc:
        logger.error("Ошибка транскрипции OpenAI: %s", str(exc), exc_info=True)
        raise TranscriptionError(f"Ошибка транскрипции аудио: {exc}") from exc

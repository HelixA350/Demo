"""
Сервис распознавания изображений через GPT-4o Vision.
"""

import base64
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.exceptions import VisionError

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


async def describe(image_bytes: bytes, media_type: str) -> str:
    """
    Описывает содержимое изображения через GPT-4o Vision.

    Args:
        image_bytes: содержимое изображения в байтах.
        media_type: MIME-тип изображения (например, 'image/jpeg').

    Returns:
        Строка с описанием изображения.

    Raises:
        VisionError: при ошибке API.
        ValueError: если тип изображения не поддерживается.
    """
    settings = get_settings()

    if media_type not in ALLOWED_IMAGE_MEDIA_TYPES:
        raise ValueError(
            f"Неподдерживаемый тип изображения: {media_type}. "
            f"Поддерживаются: {', '.join(ALLOWED_IMAGE_MEDIA_TYPES)}"
        )

    logger.info("Распознавание изображения: %s (%d байт)", media_type, len(image_bytes))

    # Конвертируем в base64 data URI
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{media_type};base64,{b64_data}"

    try:
        llm = ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
        )

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Опиши содержимое изображения подробно. Если на изображении есть текст — воспроизведи его.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                },
            ]
        )

        response = await llm.ainvoke([message])
        description = str(response.content)
        logger.info("Распознавание завершено: %d символов", len(description))
        return description

    except Exception as exc:
        logger.error("Ошибка распознавания изображения: %s", str(exc), exc_info=True)
        raise VisionError(f"Ошибка распознавания изображения: {exc}") from exc

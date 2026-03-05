"""
Менеджер контекстной памяти диалогов на базе LangChain RedisChatMessageHistory.

История хранится в Redis. При каждой операции сохранения список обрезается
до последних MEMORY_WINDOW_SIZE пар (вопрос + ответ), то есть максимум
2 * memory_window_size сообщений.
"""

import logging

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import BaseMessage

from app.config import get_settings

logger = logging.getLogger(__name__)

_HISTORY_KEY_PREFIX = "chat_history"


def _session_id(user_id: str) -> str:
    """Формирует session_id для RedisChatMessageHistory."""
    return f"{_HISTORY_KEY_PREFIX}:{user_id}"


def get_history(user_id: str) -> RedisChatMessageHistory:
    """
    Возвращает объект RedisChatMessageHistory для указанного пользователя.

    Args:
        user_id: строковый UUID пользователя.

    Returns:
        Экземпляр RedisChatMessageHistory, привязанный к сессии пользователя.
    """
    settings = get_settings()
    return RedisChatMessageHistory(
        session_id=_session_id(user_id),
        url=settings.redis_url,
        # TTL истории: 7 дней. Не связан с TTL авторизационной сессии.
        ttl=60 * 60 * 24 * 7,
    )


async def add_messages_with_trim(
    user_id: str,
    human_message: BaseMessage,
    ai_message: BaseMessage,
) -> None:
    """
    Добавляет пару сообщений (human + AI) в историю и обрезает её до window_size.

    RedisChatMessageHistory хранит сообщения в списке Redis.
    После добавления оставляем только последние 2*k сообщений
    (k пар human+AI), чтобы реализовать поведение BufferWindowMemory.

    Args:
        user_id: строковый UUID пользователя.
        human_message: сообщение пользователя.
        ai_message: ответ ассистента.
    """
    settings = get_settings()
    history = get_history(user_id)

    # Добавляем новую пару
    await history.aadd_messages([human_message, ai_message])

    # Обрезаем: оставляем последние 2*memory_window_size сообщений
    max_messages = settings.memory_window_size * 2
    messages: list[BaseMessage] = await history.aget_messages()

    if len(messages) > max_messages:
        trimmed = messages[-max_messages:]
        # Очищаем и записываем заново
        await history.aclear()
        await history.aadd_messages(trimmed)
        logger.debug(
            "История обрезана для user_id=%s: %d -> %d сообщений",
            user_id,
            len(messages),
            len(trimmed),
        )


async def get_messages(user_id: str) -> list[BaseMessage]:
    """
    Возвращает список сообщений из истории диалога пользователя.

    Args:
        user_id: строковый UUID пользователя.

    Returns:
        Список объектов BaseMessage (HumanMessage + AIMessage).
    """
    history = get_history(user_id)
    return await history.aget_messages()


async def clear_messages(user_id: str) -> None:
    """
    Полностью очищает историю диалога пользователя в Redis.

    Args:
        user_id: строковый UUID пользователя.
    """
    history = get_history(user_id)
    await history.aclear()
    logger.info("История диалога очищена для user_id=%s", user_id)

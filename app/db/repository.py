"""
CRUD-операции с базой данных. Только работа с БД, никакой бизнес-логики.
"""

import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message, User


async def create_user(
    session: AsyncSession,
    api_key_hash: str,
    api_key_prefix: str,
) -> User:
    """
    Создаёт нового пользователя в БД.

    Args:
        session: асинхронная сессия БД.
        api_key_hash: bcrypt-хеш API-ключа.
        api_key_prefix: префикс ключа для отображения.

    Returns:
        Созданный объект User.
    """
    user = User(api_key=api_key_hash, api_key_prefix=api_key_prefix)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def get_user_by_id(
    session: AsyncSession, user_id: str
) -> User | None:
    """
    Находит пользователя по его UUID.

    Args:
        session: асинхронная сессия БД.
        user_id: строковый UUID пользователя.

    Returns:
        Объект User или None если не найден.
    """
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        return None

    result = await session.execute(
        select(User).where(User.id == uid)
    )
    return result.scalar_one_or_none()


async def get_user_by_prefix(
    session: AsyncSession, prefix: str
) -> User | None:
    """
    Находит пользователя по префиксу API-ключа.

    Используется только при регистрации для проверки уникальности префикса.

    Args:
        session: асинхронная сессия БД.
        prefix: префикс ключа.

    Returns:
        Объект User или None если не найден.
    """
    result = await session.execute(
        select(User).where(User.api_key_prefix == prefix)
    )
    return result.scalar_one_or_none()


async def create_message(
    session: AsyncSession,
    user_id: uuid.UUID,
    role: str,
    content: str,
    input_type: str,
) -> Message:
    """
    Создаёт запись сообщения в БД.

    Args:
        session: асинхронная сессия БД.
        user_id: UUID пользователя-владельца.
        role: роль отправителя ('user', 'assistant', 'filtered').
        content: текст сообщения.
        input_type: тип ввода ('text', 'audio', 'image', 'audio_image', 'text_image').

    Returns:
        Созданный объект Message.
    """
    message = Message(
        user_id=user_id,
        role=role,
        content=content,
        input_type=input_type,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


async def get_user_messages(
    session: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 50,
) -> Sequence[Message]:
    """
    Возвращает историю сообщений пользователя, отсортированную по времени (новые первые).

    Args:
        session: асинхронная сессия БД.
        user_id: UUID пользователя.
        limit: максимальное количество сообщений.

    Returns:
        Список объектов Message.
    """
    result = await session.execute(
        select(Message)
        .where(Message.user_id == user_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()

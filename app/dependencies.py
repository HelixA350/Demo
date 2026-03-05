"""
Общие FastAPI зависимости: авторизация и сессия БД.
"""

from typing import AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_api_key
from app.db.base import async_session_maker
from app.db.models import User
from app.db.repository import get_user_by_prefix


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Открывает сессию БД на время запроса и гарантирует её закрытие."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def require_auth(
    x_api_key: str = Header(..., alias="X-API-Key"),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """
    Проверяет API-ключ из заголовка X-API-Key.

    Возвращает объект пользователя при успешной авторизации.
    Raises:
        HTTPException 401: если ключ не найден или хеш не совпадает.
        HTTPException 403: если пользователь неактивен.
    """
    # Извлекаем префикс (часть до точки)
    parts = x_api_key.split(".", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный формат API-ключа.",
        )

    prefix = parts[0]
    user = await get_user_by_prefix(session, prefix)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный API-ключ.",
        )

    if not verify_api_key(x_api_key, user.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный API-ключ.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Пользователь неактивен.",
        )

    return user

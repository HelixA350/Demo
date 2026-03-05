"""
Общие FastAPI зависимости: авторизация через user_id + API-ключ с Redis-кэшем.

Схема авторизации:
  - Клиент передаёт X-User-ID (UUID пользователя) и X-API-Key (секретный ключ).
  - Сначала проверяется Redis-кэш: если пара (user_id, hash) закэширована —
    верифицируем bcrypt прямо из кэша, без обращения к Postgres.
  - При cache miss — загружаем пользователя из Postgres, проверяем хеш,
    кладём хеш в Redis на auth_cache_ttl секунд.
  - TTL сессии скользящий: обновляется при каждом успешном запросе.
"""

import logging
from typing import AsyncGenerator

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_api_key
from app.core.session_cache import get_cached_hash, set_cached_hash
from app.db.base import async_session_maker
from app.db.models import User
from app.db.repository import get_user_by_id

logger = logging.getLogger(__name__)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Открывает сессию БД на время запроса и гарантирует её закрытие."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def require_auth(
    request: Request,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_api_key: str = Header(..., alias="X-API-Key"),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """
    Проверяет заголовки X-User-ID и X-API-Key.

    Алгоритм:
    1. Ищем хеш ключа в Redis по user_id (cache hit — пропускаем Postgres).
    2. При cache miss — загружаем пользователя из Postgres, кладём хеш в Redis.
    3. Верифицируем bcrypt(x_api_key, stored_hash).
    4. Проверяем is_active (только при обращении к Postgres).

    Args:
        request: FastAPI Request (для доступа к app.state.redis).
        x_user_id: UUID пользователя из заголовка X-User-ID.
        x_api_key: API-ключ из заголовка X-API-Key.
        session: сессия Postgres (используется только при cache miss).

    Returns:
        Объект User при успешной авторизации.

    Raises:
        HTTPException 401: если user_id или ключ невалидны.
        HTTPException 403: если пользователь неактивен.
    """
    redis = getattr(request.app.state, "redis", None)
    user: User | None = None

    # Пробуем Redis-кэш
    cached_hash: str | None = None
    if redis is not None:
        cached_hash = await get_cached_hash(redis, x_user_id)

    if cached_hash is not None:
        # Cache hit: верифицируем ключ без похода в Postgres
        if not verify_api_key(x_api_key, cached_hash):
            logger.warning("Неверный API-ключ (cache hit) для user_id=%s", x_user_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверный API-ключ.",
            )
        # При cache hit не проверяем is_active повторно —
        # если пользователя деактивировали, кэш протухнет через auth_cache_ttl.
        # Для немедленной блокировки можно вызвать invalidate_cached_session при деактивации.
        logger.debug("Авторизация через кэш для user_id=%s", x_user_id)
        # Возвращаем «облегчённый» User только с id — достаточно для бизнес-логики
        # (роутеры используют current_user.id для записи сообщений и работы с памятью)
        user = User(id=x_user_id)  # type: ignore[arg-type]
    else:
        # Cache miss: идём в Postgres
        user = await get_user_by_id(session, x_user_id)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Пользователь не найден.",
            )

        if not verify_api_key(x_api_key, user.api_key):
            logger.warning("Неверный API-ключ (db lookup) для user_id=%s", x_user_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверный API-ключ.",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Пользователь неактивен.",
            )

        # Кладём хеш в Redis для следующих запросов
        if redis is not None:
            await set_cached_hash(redis, x_user_id, user.api_key)

        logger.debug("Авторизация через Postgres для user_id=%s", x_user_id)

    return user

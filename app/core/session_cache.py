"""
Кэш авторизационных сессий в Redis.

Схема:
  Ключ:  auth_session:{user_id}
  Значение: bcrypt-хеш API-ключа пользователя (строка)
  TTL:   settings.auth_cache_ttl секунд (по умолчанию 15 минут)

При первом запросе с данным user_id запись создаётся из данных Postgres
и кладётся в Redis. Последующие запросы проверяют только Redis.
При обращении к существующей записи TTL автоматически обновляется (скользящее окно).

Безопасность:
  - В Redis хранится только bcrypt-хеш, а не сам ключ — даже при компрометации
    Redis злоумышленник не получает оригинальный ключ.
  - Redis не должен быть доступен публично; он работает во внутренней Docker-сети.
"""

import logging

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)

_AUTH_KEY_PREFIX = "auth_session"


def _cache_key(user_id: str) -> str:
    """Формирует ключ Redis для сессии пользователя."""
    return f"{_AUTH_KEY_PREFIX}:{user_id}"


async def get_cached_hash(redis: aioredis.Redis, user_id: str) -> str | None:
    """
    Возвращает bcrypt-хеш API-ключа из кэша Redis.

    При попадании (cache hit) обновляет TTL — реализует скользящее окно сессии.

    Args:
        redis: асинхронный Redis-клиент.
        user_id: строковый UUID пользователя.

    Returns:
        Строка с bcrypt-хешем или None если запись отсутствует.
    """
    settings = get_settings()
    key = _cache_key(user_id)
    hashed = await redis.get(key)
    if hashed is not None:
        # Обновляем TTL при каждом обращении (скользящая сессия)
        await redis.expire(key, settings.auth_cache_ttl)
        logger.debug("auth cache HIT для user_id=%s", user_id)
    else:
        logger.debug("auth cache MISS для user_id=%s", user_id)
    return hashed


async def set_cached_hash(redis: aioredis.Redis, user_id: str, hashed_key: str) -> None:
    """
    Записывает bcrypt-хеш API-ключа в Redis с заданным TTL.

    Args:
        redis: асинхронный Redis-клиент.
        user_id: строковый UUID пользователя.
        hashed_key: bcrypt-хеш для хранения.
    """
    settings = get_settings()
    key = _cache_key(user_id)
    await redis.set(key, hashed_key, ex=settings.auth_cache_ttl)
    logger.debug("auth cache SET для user_id=%s, TTL=%ds", user_id, settings.auth_cache_ttl)


async def invalidate_cached_session(redis: aioredis.Redis, user_id: str) -> None:
    """
    Удаляет запись сессии из Redis (например, при деактивации пользователя).

    Args:
        redis: асинхронный Redis-клиент.
        user_id: строковый UUID пользователя.
    """
    await redis.delete(_cache_key(user_id))
    logger.info("auth cache INVALIDATED для user_id=%s", user_id)

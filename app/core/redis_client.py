"""
Асинхронный Redis-клиент (синглтон).

Инициализируется один раз при старте приложения через lifespan
и хранится в app.state.redis.
"""

import logging

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)


async def create_redis_client() -> aioredis.Redis:
    """
    Создаёт и возвращает подключение к Redis.

    Returns:
        Экземпляр aioredis.Redis.

    Raises:
        ConnectionError: если не удалось подключиться к Redis.
    """
    settings = get_settings()
    client: aioredis.Redis = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    # Проверяем доступность
    try:
        await client.ping()
        logger.info("Подключение к Redis установлено: %s", settings.redis_url)
    except Exception as exc:
        raise ConnectionError(
            f"Не удалось подключиться к Redis по адресу '{settings.redis_url}': {exc}"
        ) from exc

    return client

"""
Эндпоинт регистрации пользователей.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import generate_api_key
from app.db.repository import create_user
from app.dependencies import get_db_session
from app.schemas import RegisterResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register", response_model=RegisterResponse)
async def register(
    session: AsyncSession = Depends(get_db_session),
) -> RegisterResponse:
    """
    Регистрирует нового пользователя и возвращает API-ключ.

    Ключ показывается только один раз. Сохраните его.
    """
    raw_key, hashed_key, prefix = generate_api_key()
    await create_user(session, api_key_hash=hashed_key, api_key_prefix=prefix)

    logger.info("Зарегистрирован новый пользователь с префиксом: %s", prefix)

    return RegisterResponse(
        api_key=raw_key,
        message="Сохраните ключ, он больше не будет показан.",
    )

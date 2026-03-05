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
    Регистрирует нового пользователя.

    Возвращает user_id и API-ключ. Сохраните оба — они используются
    как пара при каждом запросе (X-User-ID + X-API-Key).
    Ключ показывается только один раз.
    """
    raw_key, hashed_key, prefix = generate_api_key()
    user = await create_user(session, api_key_hash=hashed_key, api_key_prefix=prefix)

    logger.info("Зарегистрирован новый пользователь: id=%s, prefix=%s", user.id, prefix)

    return RegisterResponse(
        user_id=str(user.id),
        api_key=raw_key,
        message="Сохраните user_id и api_key — они больше не будут показаны вместе.",
    )

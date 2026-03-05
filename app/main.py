"""
Точка входа FastAPI приложения с lifespan для инициализации ресурсов.
"""

import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1 import auth, chat
from app.config import get_settings
from app.core.redis_client import create_redis_client
from app.exceptions import AppError
from app.vectorstore.builder import ensure_vectorstore_exists
from app.vectorstore.loader import load_vectorstore

# ─── Настройка логирования ────────────────────────────────────────────────────

LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Инициализация и завершение ресурсов приложения."""
    settings = get_settings()
    logger.info("Запуск приложения...")

    # Подключение к Redis
    redis_client = await create_redis_client()
    application.state.redis = redis_client
    logger.info("Redis подключён.")

    # Построение векторной БД если она отсутствует.
    # build_vectorstore() и load_vectorstore() синхронные (CPU/IO bound),
    # оборачиваем в to_thread чтобы не блокировать event loop.
    await asyncio.to_thread(ensure_vectorstore_exists)

    vectorstore = await asyncio.to_thread(
        load_vectorstore,
        settings.vectorstore_path,
        settings.openai_embedding_model,
    )
    application.state.vectorstore = vectorstore
    logger.info("FAISS-индекс загружен в память.")

    yield

    # Завершение работы
    await redis_client.aclose()
    logger.info("Соединение с Redis закрыто.")
    logger.info("Завершение работы приложения.")


# ─── Приложение ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Consultant API",
    version="1.0.0",
    description=(
        "ИИ-консультант на основе документа. "
        "Поддерживает текстовые, аудио и визуальные запросы. "
        "Авторизация: заголовки X-User-ID + X-API-Key."
    ),
    lifespan=lifespan,
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])


# ─── Глобальные обработчики ошибок ────────────────────────────────────────────


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Обработчик ошибок валидации Pydantic."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Обработчик кастомных ошибок приложения."""
    logger.error("AppError: %s", exc.message, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": exc.message},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Обработчик непойманных исключений."""
    logger.error("Необработанное исключение: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера."},
    )

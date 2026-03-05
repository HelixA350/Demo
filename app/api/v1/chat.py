"""
Эндпоинты чата: текстовый, аудио-запрос, очистка памяти.
"""

import logging
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import memory as memory_store
from app.db.models import User
from app.db.repository import create_message
from app.dependencies import get_db_session, require_auth
from app.schemas import ChatAudioResponse, ChatTextResponse, ClearMemoryResponse
from app.services import rag, transcription, vision

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _get_vectorstore(request: Request):
    """Извлекает FAISS-индекс из состояния приложения."""
    vs = getattr(request.app.state, "vectorstore", None)
    if vs is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Векторная база не загружена.",
        )
    return vs


async def _save_messages(
    session: AsyncSession,
    user_id: uuid.UUID,
    user_content: str,
    assistant_content: str,
    input_type: str,
    is_filtered: bool,
) -> None:
    """Фоновая задача: сохраняет пару сообщений пользователя и ассистента в Postgres."""
    await create_message(
        session,
        user_id=user_id,
        role="user",
        content=user_content,
        input_type=input_type,
    )
    await create_message(
        session,
        user_id=user_id,
        role="filtered" if is_filtered else "assistant",
        content=assistant_content,
        input_type=input_type,
    )


@router.post("/text", response_model=ChatTextResponse)
async def chat_text(
    request: Request,
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    image: UploadFile | None = File(default=None),
    current_user: User = Depends(require_auth),
    session: AsyncSession = Depends(get_db_session),
) -> ChatTextResponse:
    """
    Текстовый запрос к ИИ-консультанту. Опционально с изображением.

    Headers: X-User-ID, X-API-Key
    Body: multipart/form-data — message (str), image (file, опционально)
    """
    logger.info(
        "Запрос /chat/text: user_id=%s, has_image=%s",
        current_user.id,
        image is not None,
    )

    if not message.strip():
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым.")

    final_text = message
    input_type = "text"

    if image is not None:
        content_type = image.content_type or ""
        ext = os.path.splitext(image.filename or "")[1].lower()

        if content_type not in ALLOWED_IMAGE_CONTENT_TYPES and ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый тип изображения: {content_type}. "
                       f"Поддерживаются: jpeg, png, webp, gif.",
            )

        image_bytes = await image.read()
        image_description = await vision.describe(image_bytes, content_type or "image/jpeg")
        final_text = f"{message}\n\n[Изображение: {image_description}]"
        input_type = "text_image"

    vectorstore = _get_vectorstore(request)
    rag_response = await rag.query(
        user_id=str(current_user.id),
        user_query=final_text,
        vectorstore=vectorstore,
    )

    background_tasks.add_task(
        _save_messages,
        session,
        current_user.id,
        final_text,
        rag_response.answer,
        input_type,
        rag_response.is_filtered,
    )

    return ChatTextResponse(
        answer=rag_response.answer,
        source_chunks=rag_response.source_chunks,
    )


@router.post("/audio", response_model=ChatAudioResponse)
async def chat_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    image: UploadFile | None = File(default=None),
    current_user: User = Depends(require_auth),
    session: AsyncSession = Depends(get_db_session),
) -> ChatAudioResponse:
    """
    Аудио-запрос к ИИ-консультанту. Обязателен mp3-файл, опционально изображение.

    Headers: X-User-ID, X-API-Key
    Body: multipart/form-data — audio (mp3), image (file, опционально)
    """
    logger.info(
        "Запрос /chat/audio: user_id=%s, has_image=%s",
        current_user.id,
        image is not None,
    )

    audio_content_type = audio.content_type or ""
    audio_filename = audio.filename or "audio.mp3"
    audio_ext = os.path.splitext(audio_filename)[1].lower()

    if audio_content_type not in {"audio/mpeg", "audio/mp3"} and audio_ext != ".mp3":
        raise HTTPException(
            status_code=400,
            detail="Поддерживается только формат mp3 (audio/mpeg).",
        )

    audio_bytes = await audio.read()

    if len(audio_bytes) > transcription.MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Файл слишком большой. Максимум 25 МБ.",
        )

    transcribed_text = await transcription.transcribe(audio_bytes, audio_filename)
    final_text = transcribed_text
    input_type = "audio"

    if image is not None:
        content_type = image.content_type or ""
        ext = os.path.splitext(image.filename or "")[1].lower()

        if content_type not in ALLOWED_IMAGE_CONTENT_TYPES and ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый тип изображения: {content_type}.",
            )

        image_bytes = await image.read()
        image_description = await vision.describe(image_bytes, content_type or "image/jpeg")
        final_text = f"{transcribed_text}\n\n[Изображение: {image_description}]"
        input_type = "audio_image"

    vectorstore = _get_vectorstore(request)
    rag_response = await rag.query(
        user_id=str(current_user.id),
        user_query=final_text,
        vectorstore=vectorstore,
    )

    background_tasks.add_task(
        _save_messages,
        session,
        current_user.id,
        final_text,
        rag_response.answer,
        input_type,
        rag_response.is_filtered,
    )

    return ChatAudioResponse(
        transcription=transcribed_text,
        answer=rag_response.answer,
        source_chunks=rag_response.source_chunks,
    )


@router.delete("/memory", response_model=ClearMemoryResponse)
async def clear_memory(
    current_user: User = Depends(require_auth),
) -> ClearMemoryResponse:
    """
    Очищает историю диалога текущего пользователя в Redis.

    Headers: X-User-ID, X-API-Key
    """
    await memory_store.clear_messages(str(current_user.id))
    logger.info("Память очищена для user_id=%s", current_user.id)
    return ClearMemoryResponse(message="История диалога очищена.")

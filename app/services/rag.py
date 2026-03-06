"""
RAG-сервис: поиск по векторной базе + генерация ответа через LLM.

Пайплайн разбит на три независимых шага:
  1. retrieve()        — семантический поиск чанков в FAISS
  2. build_messages()  — сборка списка сообщений (system + история + human)
  3. generate()        — вызов LLM и сохранение пары в историю Redis
"""

import base64
import json
import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.core import memory as memory_store
from app.schemas import SourceChunk

logger = logging.getLogger(__name__)

# ─── Системный промпт ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — корпоративный ИИ-консультант для сотрудников компании «ТехноИнновации».
Твоя задача — помогать коллегам с вопросами о компании, внутренних процессах и работе в ней,
используя исключительно предоставленные документы.

Твои правила:
1. Отвечай ТОЛЬКО на основе контекста из документа.
2. Если информация по вопросу отсутствует в предоставленном контексте — честно сообщи об этом: «В документе нет информации по данному вопросу.»
3. Не придумывай и не домысливай информацию.
4. Отвечай на том же языке, на котором задан вопрос.
5. Будь точным, лаконичным и полезным.
6. Если вопрос касается насилия, оружия, самоповреждения, террора или других опасных тем — вежливо откажись отвечать и предложи задать вопрос по теме документа.

Контекст из документа (чанки пронумерованы с 0):
{context}

ФОРМАТ ОТВЕТА — СТРОГО JSON, без каких-либо пояснений вне JSON, без markdown-обёртки:
{{"content": "<текст ответа>", "used_chunk_indices": [<индексы использованных чанков>]}}"""


# ─── Схемы ────────────────────────────────────────────────────────────────────


class LLMAnswer(BaseModel):
    """Структурированный ответ LLM."""

    content: str = Field(description="Текст ответа на вопрос пользователя.")
    used_chunk_indices: list[int] = Field(
        description=(
            "Список индексов чанков (0-based), которые были фактически использованы "
            "при формировании ответа. Если ни один чанк не использовался — пустой список."
        )
    )


class RAGResponse(BaseModel):
    """Ответ RAG-сервиса."""

    content: str
    used_chunk_indices: list[int]
    source_chunks: list[SourceChunk]


# ─── Шаг 1: Retrieval ─────────────────────────────────────────────────────────


async def retrieve(user_query: str, vectorstore: FAISS, k: int = 4) -> list[SourceChunk]:
    """
    Выполняет семантический поиск в FAISS и возвращает список чанков с метаданными.

    Args:
        user_query: текстовый запрос пользователя.
        vectorstore: загруженный FAISS-индекс.
        k: количество возвращаемых чанков.

    Returns:
        Список SourceChunk с полями source, content и confidence_score.
    """
    results = await vectorstore.asimilarity_search_with_relevance_scores(user_query, k=k)

    chunks = [
        SourceChunk(
            source=os.path.basename(doc.metadata.get("source", "unknown")),
            content=doc.page_content,
            confidence_score=round(score, 4),
        )
        for doc, score in results
    ]

    logger.info(
        "Retrieval: найдено %d чанков, scores=%s для запроса %r",
        len(chunks),
        [c.confidence_score for c in chunks],
        user_query[:80],
    )
    return chunks


# ─── Шаг 2: Подготовка контекста ─────────────────────────────


async def build_messages(
    user_query: str,
    source_chunks: list[SourceChunk],
    user_id: str,
    image_bytes: bytes | None = None,
    image_media_type: str = "image/jpeg",
) -> list[BaseMessage]:
    """
    Собирает список сообщений для LLM: system prompt с пронумерованными чанками,
    история диалога пользователя из Redis, текущий human message.

    Если переданы image_bytes — формирует multipart HumanMessage с текстом и картинкой,
    которую модель получает напрямую (без предварительного распознавания).

    Args:
        user_query: текущий вопрос пользователя.
        source_chunks: чанки из FAISS, образующие контекст.
        user_id: UUID пользователя (ключ истории в Redis).
        image_bytes: байты изображения или None.
        image_media_type: MIME-тип изображения.

    Returns:
        Список BaseMessage готовый к передаче в LLM.
    """
    numbered = "\n\n---\n\n".join(
        f"[{i}] {chunk.content}" for i, chunk in enumerate(source_chunks)
    )
    system_message = SystemMessage(content=SYSTEM_PROMPT.format(context=numbered))

    chat_history = await memory_store.get_messages(user_id)

    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{image_media_type};base64,{b64}"
        human_message = HumanMessage(content=[
            {"type": "text", "text": user_query},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ])
        logger.info(
            "build_messages: добавлено изображение (%s, %d байт)",
            image_media_type, len(image_bytes),
        )
    else:
        human_message = HumanMessage(content=user_query)

    logger.debug(
        "build_messages: history=%d msgs, context_chunks=%d для user_id=%s",
        len(chat_history),
        len(source_chunks),
        user_id,
    )

    return [system_message, *chat_history, human_message]


# ─── Шаг 3: генерация ───────────────────────────────────────────────────────


def _build_llm(settings: Settings) -> ChatOpenAI:
    """Создаёт экземпляр ChatOpenAI с параметрами из настроек."""
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        max_tokens=settings.openai_max_tokens,
        temperature=0.3,
    )


async def generate(
    user_id: str,
    human_message: HumanMessage,
    messages: list[BaseMessage],
    settings: Settings,
) -> LLMAnswer:
    """
    Вызывает LLM напрямую, парсит JSON из ответа вручную,
    сохраняет пару (human + AI) в историю Redis.

    Args:
        user_id: UUID пользователя (для сохранения в память).
        human_message: сообщение пользователя (нужно для сохранения в память).
        messages: полный список сообщений для LLM (system + history + human).
        settings: объект Settings приложения.

    Returns:
        LLMAnswer со структурированным ответом модели.
    """
    llm = _build_llm(settings)

    logger.info("LLM запрос: модель=%s, user_id=%s", settings.openai_chat_model, user_id)
    raw_message = await llm.ainvoke(messages)

    raw_content: str = getattr(raw_message, "content", "") or ""
    logger.info("[RAW] content=%r", raw_content)

    # Убираем markdown-обёртку ```json ... ``` если модель её добавила
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(cleaned)
        answer = LLMAnswer(**data)
    except Exception as e:
        logger.error("Не удалось распарсить JSON: %s | cleaned=%r", e, cleaned)
        raise ValueError(f"LLM вернул невалидный JSON: {e}") from e

    # Логируем токены
    if hasattr(raw_message, "usage_metadata") and raw_message.usage_metadata:
        usage = raw_message.usage_metadata
        logger.info(
            "Токены — входные: %s, выходные: %s, всего: %s",
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )

    # Сохраняем в историю Redis
    ai_message = AIMessage(content=answer.content)
    await memory_store.add_messages_with_trim(user_id, human_message, ai_message)

    return answer


# ─── Публичный интерфейс ──────────────────────────────────────────────────────


async def query(
    user_id: str,
    user_query: str,
    vectorstore: FAISS,
    image_bytes: bytes | None = None,
    image_media_type: str = "image/jpeg",
) -> RAGResponse:
    """
    Оркестрирует полный RAG-пайплайн: retrieve -> build_messages -> generate.

    Args:
        user_id: строковый UUID пользователя.
        user_query: текстовый запрос пользователя.
        vectorstore: загруженный FAISS-индекс.
        image_bytes: байты изображения или None.
        image_media_type: MIME-тип изображения.

    Returns:
        RAGResponse с ответом модели, индексами использованных чанков и всеми чанками.
    """
    settings = get_settings()

    source_chunks = await retrieve(user_query, vectorstore)
    messages = await build_messages(
        user_query, source_chunks, user_id,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
    )

    human_message: HumanMessage = messages[-1]

    llm_answer = await generate(user_id, human_message, messages, settings)

    return RAGResponse(
        content=llm_answer.content,
        used_chunk_indices=llm_answer.used_chunk_indices,
        source_chunks=source_chunks,
    )

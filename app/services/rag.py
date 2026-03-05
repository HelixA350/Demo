"""
RAG-сервис: поиск по векторной базе + генерация ответа через LLM.

Пайплайн разбит на три независимых шага:
  1. retrieve()        — семантический поиск чанков в FAISS
  2. build_messages()  — сборка списка сообщений (system + история + human)
  3. generate()        — вызов LLM со structured output и сохранение пары в историю Redis
"""

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
{context}"""


# ─── Structured output схема ──────────────────────────────────────────────────


class LLMAnswer(BaseModel):
    """Структурированный ответ LLM."""

    content: str = Field(description="Текст ответа на вопрос пользователя.")
    used_chunk_indices: list[int] = Field(
        description=(
            "Список индексов чанков (0-based), которые были фактически использованы "
            "при формировании ответа. Если ни один чанк не использовался — пустой список."
        )
    )


# ─── Модель ответа сервиса ────────────────────────────────────────────────────


class RAGResponse(BaseModel):
    """Ответ RAG-сервиса."""

    content: str
    used_chunk_indices: list[int]
    source_chunks: list[SourceChunk]


# ─── Шаг 1: Retrieval ─────────────────────────────────────────────────────────


async def retrieve(user_query: str, vectorstore: FAISS, k: int = 4) -> list[SourceChunk]:
    """
    Выполняет семантический поиск в FAISS и возвращает список чанков с метаданными.

    Поле source берётся из doc.metadata["source"], которое LangChain-загрузчики
    (PyPDFLoader, TextLoader) проставляют автоматически как путь к исходному файлу.
    В ответе оставляем только имя файла без пути — удобнее для клиента.

    confidence_score вычисляется через similarity_search_with_relevance_scores:
    LangChain нормализует скор в [0, 1] с учётом типа метрики индекса (L2 или cosine),
    где 1.0 = идеальное совпадение.
    Скор округляется до 4 знаков после запятой.

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


# ─── Шаг 2: Context enrichment & prompt assembly ─────────────────────────────


async def build_messages(
    user_query: str,
    source_chunks: list[SourceChunk],
    user_id: str,
) -> list[BaseMessage]:
    """
    Собирает список сообщений для LLM: system prompt с пронумерованными чанками,
    история диалога пользователя из Redis, текущий human message.

    Чанки нумеруются с 0 — модель ссылается на них по индексу в used_chunk_indices.

    Args:
        user_query: текущий вопрос пользователя.
        source_chunks: чанки из FAISS, образующие контекст.
        user_id: UUID пользователя (ключ истории в Redis).

    Returns:
        Список BaseMessage готовый к передаче в LLM.
    """
    numbered = "\n\n---\n\n".join(
        f"[{i}] {chunk.content}" for i, chunk in enumerate(source_chunks)
    )
    system_message = SystemMessage(content=SYSTEM_PROMPT.format(context=numbered))

    chat_history = await memory_store.get_messages(user_id)
    human_message = HumanMessage(content=user_query)

    logger.debug(
        "build_messages: history=%d msgs, context_chunks=%d для user_id=%s",
        len(chat_history),
        len(source_chunks),
        user_id,
    )

    return [system_message, *chat_history, human_message]


# ─── Шаг 3: Generation ───────────────────────────────────────────────────────


def _build_llm(settings: Settings) -> ChatOpenAI:
    """
    Создаёт экземпляр ChatOpenAI с параметрами из настроек.

    Args:
        settings: объект Settings приложения.

    Returns:
        Настроенный экземпляр ChatOpenAI.
    """
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        max_tokens=settings.openai_max_tokens,
    )


async def generate(
    user_id: str,
    human_message: HumanMessage,
    messages: list[BaseMessage],
    settings: Settings,
) -> LLMAnswer:
    """
    Вызывает LLM через with_structured_output, логирует токены
    и сохраняет пару (human + AI) в историю Redis.

    Args:
        user_id: UUID пользователя (для сохранения в память).
        human_message: сообщение пользователя (нужно для сохранения в память).
        messages: полный список сообщений для LLM (system + history + human).
        settings: объект Settings приложения.

    Returns:
        LLMAnswer со структурированным ответом модели.
    """
    llm = _build_llm(settings).with_structured_output(LLMAnswer, include_raw=True)

    logger.info("LLM запрос: модель=%s, user_id=%s", settings.openai_chat_model, user_id)
    raw = await llm.ainvoke(messages)

    # include_raw=True возвращает dict: {"raw": AIMessage, "parsed": LLMAnswer, "parsing_error": ...}
    answer: LLMAnswer = raw["parsed"]

    if raw.get("parsing_error"):
        logger.warning("Ошибка парсинга structured output: %s", raw["parsing_error"])

    # Логируем токены из raw AIMessage
    raw_message = raw.get("raw")
    if raw_message and hasattr(raw_message, "usage_metadata") and raw_message.usage_metadata:
        usage = raw_message.usage_metadata
        logger.info(
            "Токены — входные: %s, выходные: %s, всего: %s",
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )

    # Сохраняем в историю Redis — используем content как текст ответа
    ai_message = AIMessage(content=answer.content)
    await memory_store.add_messages_with_trim(user_id, human_message, ai_message)

    return answer


# ─── Публичный интерфейс ──────────────────────────────────────────────────────


async def query(user_id: str, user_query: str, vectorstore: FAISS) -> RAGResponse:
    """
    Оркестрирует полный RAG-пайплайн: retrieve -> build_messages -> generate.

    Args:
        user_id: строковый UUID пользователя.
        user_query: текстовый запрос (уже с описанием изображения/транскрипцией если нужно).
        vectorstore: загруженный FAISS-индекс.

    Returns:
        RAGResponse с ответом модели, индексами использованных чанков и всеми чанками.
    """
    settings = get_settings()

    source_chunks = await retrieve(user_query, vectorstore)
    messages = await build_messages(user_query, source_chunks, user_id)

    human_message: HumanMessage = messages[-1]

    llm_answer = await generate(user_id, human_message, messages, settings)

    return RAGResponse(
        content=llm_answer.content,
        used_chunk_indices=llm_answer.used_chunk_indices,
        source_chunks=source_chunks,
    )

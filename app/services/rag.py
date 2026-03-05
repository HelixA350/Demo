"""
RAG-сервис: поиск по векторной базе + генерация ответа через LLM.

Пайплайн разбит на три независимых шага:
  1. retrieve()        — семантический поиск чанков в FAISS
  2. build_messages()  — сборка списка сообщений (system + история + human)
  3. generate()        — вызов LLM и сохранение пары в историю Redis
"""

import logging

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.core import memory as memory_store

logger = logging.getLogger(__name__)

# ─── Системный промпт ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — ИИ-консультант, который отвечает на вопросы исключительно на основе предоставленного документа.

Твои правила:
1. Отвечай ТОЛЬКО на основе контекста из документа.
2. Если информация по вопросу отсутствует в предоставленном контексте — честно сообщи об этом: «В документе нет информации по данному вопросу.»
3. Не придумывай и не домысливай информацию.
4. Отвечай на том же языке, на котором задан вопрос.
5. Будь точным, лаконичным и полезным.
6. Если вопрос касается насилия, оружия, самоповреждения, террора или других опасных тем — вежливо откажись отвечать и предложи задать вопрос по теме документа.

Контекст из документа:
{context}"""


# ─── Модель ответа ────────────────────────────────────────────────────────────


class RAGResponse(BaseModel):
    """Ответ RAG-сервиса."""

    answer: str
    source_chunks: list[str]


# ─── Шаг 1: Retrieval ─────────────────────────────────────────────────────────


async def retrieve(user_query: str, vectorstore: FAISS, k: int = 4) -> list[str]:
    """
    Выполняет семантический поиск в FAISS и возвращает список текстовых чанков.

    Args:
        user_query: текстовый запрос пользователя.
        vectorstore: загруженный FAISS-индекс.
        k: количество возвращаемых чанков.

    Returns:
        Список строк — содержимое найденных документов.
    """
    docs = await vectorstore.asimilarity_search(user_query, k=k)
    chunks = [doc.page_content for doc in docs]
    logger.info("Retrieval: найдено %d чанков для запроса %r", len(chunks), user_query[:80])
    return chunks


# ─── Шаг 2: Подготовка контекста ─────────────────────────────


async def build_messages(
    user_query: str,
    source_chunks: list[str],
    user_id: str,
) -> list[BaseMessage]:
    """
    Собирает список сообщений для LLM: system prompt с контекстом,
    история диалога пользователя из Redis, текущий human message.

    Args:
        user_query: текущий вопрос пользователя.
        source_chunks: чанки из FAISS, образующие контекст.
        user_id: UUID пользователя (ключ истории в Redis).

    Returns:
        Список BaseMessage готовый к передаче в LLM.
    """
    context = "\n\n---\n\n".join(source_chunks)
    system_message = SystemMessage(content=SYSTEM_PROMPT.format(context=context))

    chat_history = await memory_store.get_messages(user_id)
    human_message = HumanMessage(content=user_query)

    logger.debug(
        "build_messages: history=%d msgs, context_chunks=%d для user_id=%s",
        len(chat_history),
        len(source_chunks),
        user_id,
    )

    return [system_message, *chat_history, human_message]


# ─── Шаг 3: Генерация ───────────────────────────────────────────────────────


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
) -> str:
    """
    Вызывает LLM, логирует использование токенов и сохраняет пару
    (human + AI) в историю Redis.

    Args:
        user_id: UUID пользователя (для сохранения в память).
        human_message: сообщение пользователя (нужно для сохранения в память).
        messages: полный список сообщений для LLM (system + history + human).
        settings: объект Settings приложения.

    Returns:
        Строка с ответом модели.
    """
    llm = _build_llm(settings)

    logger.info("LLM запрос: модель=%s, user_id=%s", settings.openai_chat_model, user_id)
    response = await llm.ainvoke(messages)
    answer = str(response.content)

    # Логируем токены
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        logger.info(
            "Токены — входные: %s, выходные: %s, всего: %s",
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )

    # Сохраняем пару в Redis
    ai_message = AIMessage(content=answer)
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
        RAGResponse с ответом модели и исходными чанками.
    """
    settings = get_settings()

    source_chunks = await retrieve(user_query, vectorstore)
    messages = await build_messages(user_query, source_chunks, user_id)

    # Последний элемент списка — всегда HumanMessage текущего запроса
    human_message: HumanMessage = messages[-1]  # type: ignore[assignment]

    answer = await generate(user_id, human_message, messages, settings)

    return RAGResponse(answer=answer, source_chunks=source_chunks)

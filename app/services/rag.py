"""
RAG-сервис: поиск по векторной базе + генерация ответа через LLM.
"""

import logging
import re

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import get_settings
from app.core import memory as memory_store

logger = logging.getLogger(__name__)

# ─── Фильтрация запрещённых тематик ──────────────────────────────────────────

FORBIDDEN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(смерт[ьи]|умер|умрёт|умрет|убит|убийств|суицид|самоубийств)\b", re.IGNORECASE),
    re.compile(r"\b(насили[еяй]|насилов|изнасил)\b", re.IGNORECASE),
    re.compile(r"\b(самоповрежден|порез.*себ|вред.*себ|harm.*self|self.?harm)\b", re.IGNORECASE),
    re.compile(r"\b(оружи[ея]|взрывчат|бомб[аы]|пистолет|автомат|rifle|gun|bomb|weapon)\b", re.IGNORECASE),
    re.compile(r"\b(террор|терракт|джихад|экстремизм|terror|terrorism)\b", re.IGNORECASE),
    re.compile(r"\b(kill|murder|suicide|death|die|dying|slaughter)\b", re.IGNORECASE),
    re.compile(r"\b(violence|violent|assault|rape|abuse)\b", re.IGNORECASE),
]

REFUSAL_RESPONSE = (
    "Я не могу ответить на этот вопрос, так как он касается запрещённой тематики. "
    "Пожалуйста, задайте другой вопрос по теме документа."
)

SYSTEM_PROMPT = """Ты — ИИ-консультант, который отвечает на вопросы исключительно на основе предоставленного документа.

Твои правила:
1. Отвечай ТОЛЬКО на основе контекста из документа.
2. Если информация по вопросу отсутствует в предоставленном контексте — честно сообщи об этом: «В документе нет информации по данному вопросу.»
3. Не придумывай и не домысливай информацию.
4. Отвечай на том же языке, на котором задан вопрос.
5. Будь точным, лаконичным и полезным.

Контекст из документа:
{context}"""


class RAGResponse(BaseModel):
    """Ответ RAG-сервиса."""

    answer: str
    source_chunks: list[str]
    is_filtered: bool = False


def _is_forbidden(query: str) -> bool:
    """Проверяет запрос на запрещённые тематики."""
    return any(pattern.search(query) for pattern in FORBIDDEN_PATTERNS)


async def query(user_id: str, user_query: str, vectorstore: FAISS) -> RAGResponse:
    """
    Выполняет RAG-запрос: поиск по векторной базе + генерация ответа.

    Args:
        user_id: строковый UUID пользователя (ключ памяти в Redis).
        user_query: текстовый запрос пользователя.
        vectorstore: загруженный FAISS-индекс.

    Returns:
        Объект RAGResponse с ответом и источниками.
    """
    settings = get_settings()

    # Проверяем на запрещённые темы
    if _is_forbidden(user_query):
        logger.warning("Запрос отфильтрован для user_id=%s: %s", user_id, user_query[:100])
        return RAGResponse(
            answer=REFUSAL_RESPONSE,
            source_chunks=[],
            is_filtered=True,
        )

    # Поиск релевантных чанков
    docs = await vectorstore.asimilarity_search(user_query, k=4)
    source_chunks = [doc.page_content for doc in docs]
    logger.info("Найдено %d чанков для user_id=%s", len(source_chunks), user_id)

    # Формируем контекст из найденных чанков
    context = "\n\n---\n\n".join(source_chunks)

    # Загружаем историю диалога из Redis
    chat_history = await memory_store.get_messages(user_id)

    # Формируем список сообщений для LLM
    system_message = SystemMessage(content=SYSTEM_PROMPT.format(context=context))
    human_message = HumanMessage(content=user_query)
    messages = [system_message] + chat_history + [human_message]

    # Вызываем LLM
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
    )

    logger.info("Запрос к модели %s для user_id=%s", settings.openai_chat_model, user_id)
    response = await llm.ainvoke(messages)
    answer = str(response.content)

    # Логируем использование токенов
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        logger.info(
            "Токены — входные: %s, выходные: %s, всего: %s",
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )

    # Сохраняем пару сообщений в Redis с обрезкой до window_size
    ai_message = AIMessage(content=answer)
    await memory_store.add_messages_with_trim(user_id, human_message, ai_message)

    return RAGResponse(answer=answer, source_chunks=source_chunks)

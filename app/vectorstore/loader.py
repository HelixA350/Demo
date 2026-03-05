"""
Загрузка FAISS-индекса в память при старте приложения.
"""

import logging

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.exceptions import VectorstoreNotLoadedError

logger = logging.getLogger(__name__)


def load_vectorstore(vectorstore_path: str, embedding_model: str) -> FAISS:
    """
    Загружает FAISS-индекс с диска в память.

    Args:
        vectorstore_path: путь к директории с сохранённым индексом.
        embedding_model: название модели эмбеддингов OpenAI.

    Returns:
        Загруженный объект FAISS.

    Raises:
        VectorstoreNotLoadedError: если файлы индекса не найдены или произошла ошибка загрузки.
    """
    settings = get_settings()

    logger.info("Загрузка FAISS-индекса из: %s", vectorstore_path)

    try:
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=settings.openai_api_key,
        )
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS-индекс загружен успешно.")
        return vectorstore
    except Exception as exc:
        raise VectorstoreNotLoadedError(
            f"Не удалось загрузить FAISS-индекс из '{vectorstore_path}'. "
            f"Убедитесь, что вы запустили скрипт индексации: python -m app.vectorstore.builder. "
            f"Ошибка: {exc}"
        ) from exc

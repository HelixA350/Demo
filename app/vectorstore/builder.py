"""
Скрипт индексации документа: чанкование, создание эмбеддингов, сохранение FAISS-индекса.

Запуск вручную: python -m app.vectorstore.builder
При старте приложения вызывается автоматически через ensure_vectorstore_exists().
"""

import logging
import os
import sys

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Файл, по наличию которого определяем что индекс уже построен
_INDEX_MARKER = "index.faiss"


def vectorstore_exists(vectorstore_path: str) -> bool:
    """
    Проверяет, существует ли уже построенный FAISS-индекс на диске.

    Args:
        vectorstore_path: путь к директории с индексом.

    Returns:
        True если индекс найден, False если нужно строить.
    """
    return os.path.isfile(os.path.join(vectorstore_path, _INDEX_MARKER))


def build_vectorstore() -> None:
    """
    Загружает документ, разбивает на чанки, создаёт FAISS-индекс и сохраняет на диск.

    Raises:
        FileNotFoundError: если документ не найден по DOCUMENT_PATH.
        ValueError: если формат документа не поддерживается.
    """
    settings = get_settings()
    doc_path = settings.document_path
    vs_path = settings.vectorstore_path

    logger.info("Загрузка документа: %s", doc_path)

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Документ не найден: {doc_path}")

    ext = os.path.splitext(doc_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(doc_path)
    elif ext == ".txt":
        loader = TextLoader(doc_path, encoding="utf-8")
    else:
        raise ValueError(
            f"Неподдерживаемый формат документа: {ext}. Поддерживаются: .pdf, .txt"
        )

    documents = loader.load()
    logger.info("Документ загружен: %d страниц/блоков", len(documents))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    logger.info("Документ разбит на %d чанков", len(chunks))

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )

    logger.info("Создание FAISS-индекса...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(vs_path, exist_ok=True)
    vectorstore.save_local(vs_path)
    logger.info("FAISS-индекс сохранён в: %s (%d чанков)", vs_path, len(chunks))


def ensure_vectorstore_exists() -> None:
    """
    Проверяет наличие FAISS-индекса и строит его если он отсутствует.

    Предназначена для вызова при старте приложения.
    Если индекс уже есть — ничего не делает.
    """
    settings = get_settings()

    if vectorstore_exists(settings.vectorstore_path):
        logger.info(
            "FAISS-индекс найден в '%s', построение пропущено.",
            settings.vectorstore_path,
        )
        return

    logger.info(
        "FAISS-индекс не найден в '%s', запускаю построение...",
        settings.vectorstore_path,
    )
    build_vectorstore()


if __name__ == "__main__":
    build_vectorstore()

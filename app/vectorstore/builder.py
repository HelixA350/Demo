"""
Скрипт индексации документа: чанкование, создание эмбеддингов, сохранение FAISS-индекса.

Запуск: python -m app.vectorstore.builder
"""

import logging
import os
import sys

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_vectorstore() -> None:
    """
    Загружает документ, разбивает на чанки, создаёт FAISS-индекс и сохраняет на диск.
    """
    settings = get_settings()
    doc_path = settings.document_path
    vs_path = settings.vectorstore_path

    logger.info("Загрузка документа: %s", doc_path)

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Документ не найден: {doc_path}")

    # Выбираем загрузчик по расширению
    ext = os.path.splitext(doc_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(doc_path)
    elif ext == ".txt":
        loader = TextLoader(doc_path, encoding="utf-8")
    else:
        raise ValueError(f"Неподдерживаемый формат документа: {ext}. Поддерживаются: .pdf, .txt")

    documents = loader.load()
    logger.info("Документ загружен: %d страниц/блоков", len(documents))

    # Разбиваем на чанки
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    logger.info("Документ разбит на %d чанков", len(chunks))

    # Создаём эмбеддинги
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )

    # Создаём FAISS-индекс
    logger.info("Создание FAISS-индекса...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Сохраняем на диск
    os.makedirs(vs_path, exist_ok=True)
    vectorstore.save_local(vs_path)
    logger.info("FAISS-индекс сохранён в: %s", vs_path)


if __name__ == "__main__":
    build_vectorstore()

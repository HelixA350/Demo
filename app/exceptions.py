"""
Кастомные исключения приложения.
"""


class AppError(Exception):
    """Базовое исключение приложения."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class TranscriptionError(AppError):
    """Ошибка транскрипции аудио через Whisper."""


class VisionError(AppError):
    """Ошибка распознавания изображения через GPT-4o Vision."""


class VectorstoreNotLoadedError(AppError):
    """Ошибка загрузки FAISS-индекса."""

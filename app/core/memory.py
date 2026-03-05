"""
Менеджер контекстной памяти диалогов на базе LangChain ConversationBufferWindowMemory.
"""

from langchain.memory import ConversationBufferWindowMemory

from app.config import get_settings


class MemoryManager:
    """
    Синглтон-менеджер памяти пользователей.

    Хранит словарь {user_id: ConversationBufferWindowMemory} в оперативной памяти.
    При перезапуске сервиса память очищается.
    """

    _instance: "MemoryManager | None" = None
    _memories: dict[str, ConversationBufferWindowMemory]

    def __new__(cls) -> "MemoryManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._memories = {}
        return cls._instance

    def get_memory(self, user_id: str) -> ConversationBufferWindowMemory:
        """
        Возвращает память для указанного пользователя.
        При первом обращении создаёт новый экземпляр памяти.

        Args:
            user_id: идентификатор пользователя.

        Returns:
            Экземпляр ConversationBufferWindowMemory.
        """
        if user_id not in self._memories:
            settings = get_settings()
            self._memories[user_id] = ConversationBufferWindowMemory(
                k=settings.memory_window_size,
                return_messages=True,
                memory_key="chat_history",
            )
        return self._memories[user_id]

    def clear_memory(self, user_id: str) -> None:
        """
        Удаляет память пользователя из словаря.

        Args:
            user_id: идентификатор пользователя.
        """
        self._memories.pop(user_id, None)


# Глобальный экземпляр менеджера памяти
memory_manager = MemoryManager()

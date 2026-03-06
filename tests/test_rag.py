"""
Тесты для app.services.rag: парсинг JSON-ответа LLM и сборка human message.
"""

import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.rag import LLMAnswer, RAGResponse
from app.schemas import SourceChunk


# ─── Фикстуры ────────────────────────────────────────────────────────────────


def make_chunks(n: int = 3) -> list[SourceChunk]:
    return [
        SourceChunk(source=f"doc_{i}.pdf", content=f"Содержимое чанка {i}", confidence_score=0.9 - i * 0.1)
        for i in range(n)
    ]


# ─── Тесты LLMAnswer ─────────────────────────────────────────────────────────


class TestLLMAnswer:
    def test_valid_json_parses(self):
        data = {"content": "Ответ модели", "used_chunk_indices": [0, 2]}
        answer = LLMAnswer(**data)
        assert answer.content == "Ответ модели"
        assert answer.used_chunk_indices == [0, 2]

    def test_empty_indices_allowed(self):
        answer = LLMAnswer(content="Текст", used_chunk_indices=[])
        assert answer.used_chunk_indices == []

    def test_missing_content_raises(self):
        with pytest.raises(Exception):
            LLMAnswer(used_chunk_indices=[0])

    def test_missing_indices_raises(self):
        with pytest.raises(Exception):
            LLMAnswer(content="Текст")


# ─── Тесты парсинга JSON из сырого ответа LLM ────────────────────────────────


class TestJsonParsing:
    """Тестируем логику парсинга которая живёт в generate()."""

    def _parse(self, raw_content: str) -> LLMAnswer:
        """Воспроизводим логику парсинга из generate()."""
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        return LLMAnswer(**data)

    def test_plain_json(self):
        raw = '{"content": "Ответ", "used_chunk_indices": [0]}'
        answer = self._parse(raw)
        assert answer.content == "Ответ"
        assert answer.used_chunk_indices == [0]

    def test_json_with_markdown_wrapper(self):
        raw = '```json\n{"content": "Ответ", "used_chunk_indices": [1, 2]}\n```'
        answer = self._parse(raw)
        assert answer.content == "Ответ"
        assert answer.used_chunk_indices == [1, 2]

    def test_json_with_whitespace(self):
        raw = '  \n  {"content": "Пробелы", "used_chunk_indices": []}  \n  '
        answer = self._parse(raw)
        assert answer.content == "Пробелы"

    def test_plain_text_raises(self):
        with pytest.raises(json.JSONDecodeError):
            self._parse("Это просто текст без JSON")

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            self._parse("")

    def test_cyrillic_content_parses(self):
        raw = '{"content": "Генеральный директор — Соколов Д.И.", "used_chunk_indices": [0, 1, 3]}'
        answer = self._parse(raw)
        assert "Соколов" in answer.content
        assert len(answer.used_chunk_indices) == 3


# ─── Тесты build_messages ────────────────────────────────────────────────────


class TestBuildMessages:
    @pytest.mark.asyncio
    async def test_text_only_message(self):
        from app.services.rag import build_messages
        from langchain_core.messages import HumanMessage, SystemMessage

        chunks = make_chunks(2)

        with patch("app.services.rag.memory_store.get_messages", new_callable=AsyncMock) as mock_mem:
            mock_mem.return_value = []
            messages = await build_messages(
                user_query="Тестовый вопрос",
                source_chunks=chunks,
                user_id="test-user-id",
            )

        assert isinstance(messages[0], SystemMessage)
        human = messages[-1]
        assert isinstance(human, HumanMessage)
        assert isinstance(human.content, str)
        assert "Тестовый вопрос" in human.content

    @pytest.mark.asyncio
    async def test_message_with_image(self):
        from app.services.rag import build_messages
        from langchain_core.messages import HumanMessage

        chunks = make_chunks(1)
        fake_image = b"\x89PNG\r\n" + b"\x00" * 100

        with patch("app.services.rag.memory_store.get_messages", new_callable=AsyncMock) as mock_mem:
            mock_mem.return_value = []
            messages = await build_messages(
                user_query="Что на картинке?",
                source_chunks=chunks,
                user_id="test-user-id",
                image_bytes=fake_image,
                image_media_type="image/png",
            )

        human = messages[-1]
        assert isinstance(human, HumanMessage)
        assert isinstance(human.content, list)
        types = [part["type"] for part in human.content]
        assert "text" in types
        assert "image_url" in types

    @pytest.mark.asyncio
    async def test_image_url_contains_base64(self):
        from app.services.rag import build_messages

        chunks = make_chunks(1)
        fake_image = b"fakeimagedata"

        with patch("app.services.rag.memory_store.get_messages", new_callable=AsyncMock) as mock_mem:
            mock_mem.return_value = []
            messages = await build_messages(
                user_query="Вопрос",
                source_chunks=chunks,
                user_id="test-user-id",
                image_bytes=fake_image,
                image_media_type="image/jpeg",
            )

        human = messages[-1]
        image_part = next(p for p in human.content if p["type"] == "image_url")
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        b64_data = url.split(",", 1)[1]
        assert base64.b64decode(b64_data) == fake_image

    @pytest.mark.asyncio
    async def test_system_prompt_contains_chunks(self):
        from app.services.rag import build_messages
        from langchain_core.messages import SystemMessage

        chunks = make_chunks(2)

        with patch("app.services.rag.memory_store.get_messages", new_callable=AsyncMock) as mock_mem:
            mock_mem.return_value = []
            messages = await build_messages(
                user_query="Вопрос",
                source_chunks=chunks,
                user_id="test-user-id",
            )

        system = messages[0]
        assert isinstance(system, SystemMessage)
        assert "[0]" in system.content
        assert "[1]" in system.content
        assert chunks[0].content in system.content

    @pytest.mark.asyncio
    async def test_chat_history_included(self):
        from app.services.rag import build_messages
        from langchain_core.messages import HumanMessage, AIMessage

        chunks = make_chunks(1)
        history = [
            HumanMessage(content="Предыдущий вопрос"),
            AIMessage(content="Предыдущий ответ"),
        ]

        with patch("app.services.rag.memory_store.get_messages", new_callable=AsyncMock) as mock_mem:
            mock_mem.return_value = history
            messages = await build_messages(
                user_query="Новый вопрос",
                source_chunks=chunks,
                user_id="test-user-id",
            )

        assert len(messages) == 4
        assert messages[1].content == "Предыдущий вопрос"
        assert messages[2].content == "Предыдущий ответ"

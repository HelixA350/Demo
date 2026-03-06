"""
Тесты для app.db.repository: CRUD-операции с БД через мок-сессию.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.db.repository import create_user, get_user_by_id, create_message
from app.db.models import User, Message


def make_mock_session():
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


class TestCreateUser:
    @pytest.mark.asyncio
    async def test_creates_user_with_correct_fields(self):
        session = make_mock_session()

        async def fake_refresh(obj):
            obj.id = uuid.uuid4()

        session.refresh.side_effect = fake_refresh
        await create_user(session, api_key_hash="hashed", api_key_prefix="ak_12345")

        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.refresh.assert_called_once()

        added_user = session.add.call_args[0][0]
        assert added_user.api_key == "hashed"
        assert added_user.api_key_prefix == "ak_12345"

    @pytest.mark.asyncio
    async def test_returns_user_instance(self):
        session = make_mock_session()
        user = await create_user(session, api_key_hash="h", api_key_prefix="ak_aaaaa")
        assert isinstance(user, User)


class TestGetUserById:
    @pytest.mark.asyncio
    async def test_returns_user_when_found(self):
        session = make_mock_session()
        user_id = uuid.uuid4()
        expected_user = User(id=user_id, api_key="hash", api_key_prefix="ak_12345")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_user
        session.execute = AsyncMock(return_value=mock_result)

        result = await get_user_by_id(session, str(user_id))
        assert result == expected_user

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        session = make_mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        result = await get_user_by_id(session, str(uuid.uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_invalid_uuid(self):
        session = make_mock_session()
        result = await get_user_by_id(session, "not-a-uuid")
        assert result is None
        session.execute.assert_not_called()


class TestCreateMessage:
    @pytest.mark.asyncio
    async def test_creates_message_with_correct_fields(self):
        session = make_mock_session()
        user_id = uuid.uuid4()
        await create_message(
            session,
            user_id=user_id,
            role="user",
            content="Тестовый вопрос",
            input_type="text",
        )

        session.add.assert_called_once()
        added = session.add.call_args[0][0]
        assert isinstance(added, Message)
        assert added.user_id == user_id
        assert added.role == "user"
        assert added.content == "Тестовый вопрос"
        assert added.input_type == "text"

    @pytest.mark.asyncio
    async def test_commits_and_refreshes(self):
        session = make_mock_session()
        await create_message(
            session,
            user_id=uuid.uuid4(),
            role="assistant",
            content="Ответ",
            input_type="text",
        )
        session.commit.assert_called_once()
        session.refresh.assert_called_once()

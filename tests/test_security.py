"""
Тесты для app.core.security: генерация и верификация API-ключей.
"""

import pytest
from app.core.security import generate_api_key, verify_api_key


class TestGenerateApiKey:
    def test_returns_three_values(self):
        result = generate_api_key()
        assert len(result) == 3

    def test_raw_key_format(self):
        raw_key, _, prefix = generate_api_key()
        # Формат: "ak_XXXXX.YYYYYYYY..."
        assert raw_key.startswith("ak_")
        assert "." in raw_key
        parts = raw_key.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 32

    def test_prefix_matches_raw_key(self):
        raw_key, _, prefix = generate_api_key()
        assert raw_key.startswith(prefix)

    def test_prefix_length(self):
        _, _, prefix = generate_api_key()
        assert len(prefix) == 8

    def test_hashed_key_is_bcrypt(self):
        _, hashed, _ = generate_api_key()
        assert hashed.startswith("$2b$")

    def test_each_call_produces_unique_key(self):
        raw1, _, _ = generate_api_key()
        raw2, _, _ = generate_api_key()
        assert raw1 != raw2

    def test_each_call_produces_unique_hash(self):
        _, hash1, _ = generate_api_key()
        _, hash2, _ = generate_api_key()
        assert hash1 != hash2


class TestVerifyApiKey:
    def test_correct_key_returns_true(self):
        raw_key, hashed, _ = generate_api_key()
        assert verify_api_key(raw_key, hashed) is True

    def test_wrong_key_returns_false(self):
        _, hashed, _ = generate_api_key()
        assert verify_api_key("ak_wrong.wrongwrongwrongwrongwrongwrong", hashed) is False

    def test_empty_key_returns_false(self):
        _, hashed, _ = generate_api_key()
        assert verify_api_key("", hashed) is False

    def test_cross_keys_return_false(self):
        raw1, _, _ = generate_api_key()
        _, hash2, _ = generate_api_key()
        assert verify_api_key(raw1, hash2) is False

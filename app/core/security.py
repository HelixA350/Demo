"""
Генерация и проверка API-ключей с использованием bcrypt.
"""

import secrets
import string

import bcrypt


def generate_api_key() -> tuple[str, str, str]:
    """
    Генерирует новый API-ключ.

    Returns:
        Tuple (raw_key, hashed_key, prefix):
            - raw_key: полный ключ для показа пользователю (один раз)
            - hashed_key: bcrypt-хеш для хранения в БД
            - prefix: первые 8 символов до точки для поиска в БД
    """
    # Генерируем префикс: "ak_" + 5 случайных hex-символов = 8 символов
    prefix_random = secrets.token_hex(4)  # 8 hex chars
    prefix = f"ak_{prefix_random[:5]}"  # итого 8 символов

    # Генерируем секретную часть
    alphabet = string.ascii_letters + string.digits
    secret_part = "".join(secrets.choice(alphabet) for _ in range(32))

    raw_key = f"{prefix}.{secret_part}"

    # Хешируем
    hashed = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()

    return raw_key, hashed, prefix


def verify_api_key(raw_key: str, hashed_key: str) -> bool:
    """
    Проверяет соответствие сырого ключа его bcrypt-хешу.

    Args:
        raw_key: ключ, переданный пользователем.
        hashed_key: хеш из БД.

    Returns:
        True если ключ совпадает, иначе False.
    """
    return bcrypt.checkpw(raw_key.encode(), hashed_key.encode())

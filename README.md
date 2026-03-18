# AI Consultant — RAG Backend API

FastAPI-сервис с RAG-архитектурой для ответов на вопросы по корпоративным регламентам. Поддерживает текстовые, аудио и мультимодальные запросы. Написан как демонстрация навыков проектирования LLM-приложений production-уровня.

**Стек:** FastAPI · PostgreSQL · Redis · FAISS · LangChain · OpenAI (GPT-4o, Whisper, Embeddings) · Docker · Alembic · Aiogram 3

---

## Что реализовано

### RAG-пайплайн
Документы из папки `data/` индексируются через `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100) и сохраняются в FAISS. При старте приложения индекс загружается в память через lifespan FastAPI — к первому запросу он уже готов. На каждый запрос из FAISS достаётся топ-N чанков, которые передаются в промпт GPT-4o. В ответе возвращается не только текст, но и список реально использованных чанков с confidence score — для отладки качества retrieval.

### Мультимодальность
- **Текст + изображение** — `POST /api/v1/chat/text`, изображение опционально, обрабатывается через GPT-4o Vision
- **Аудио + изображение** — `POST /api/v1/chat/audio`, аудио транскрибируется через Whisper, затем проходит тот же RAG-пайплайн

### Диалоговая память
История последних 10 сообщений каждого пользователя хранится в Redis. При каждом запросе LangChain читает историю и добавляет в контекст промпта. Старые сообщения вытесняются автоматически.

### Авторизация по API-ключу
При регистрации генерируется случайный 32-байтный ключ. В БД хранится только bcrypt-хеш и короткий префикс. Ключ показывается пользователю один раз. Кэш авторизационных сессий хранится в Redis с TTL 15 минут и скользящим окном — PostgreSQL не задействован при повторных запросах в рамках сессии.

### Асинхронная запись диалогов
Все сообщения пишутся в PostgreSQL через SQLAlchemy 2 + asyncpg. Запись выполняется через `BackgroundTasks` FastAPI — ответ уже отправлен клиенту, I/O базы не блокирует критический путь.

---

## Архитектура

```
app/
├── api/v1/
│   ├── auth.py          # POST /register
│   └── chat.py          # POST /text, POST /audio, DELETE /memory
├── core/
│   ├── security.py      # Генерация ключей, bcrypt
│   ├── memory.py        # Диалоговая память LangChain (окно 10 сообщений)
│   ├── redis_client.py  # Синглтон Redis-клиента
│   └── session_cache.py # Кэш сессий (TTL 15 мин, скользящее окно)
├── db/
│   ├── models.py        # ORM: User, Message
│   └── repository.py    # CRUD-функции
│   └── base.py    # Инициализация SQLAlchemy
├── services/
│   ├── rag.py           # RAG-цепочка: FAISS → GPT-4o
│   ├── transcription.py # Whisper
│   └── vision.py        # GPT-4o Vision
└── vectorstore/
    ├── builder.py       # Индексация: чанкование → эмбеддинги → FAISS
    └── loader.py        # Загрузка индекса при старте
```

---

## Быстрый старт

**1. Переменные окружения**

```bash
cp .env.example .env
# Заполните OPENAI_API_KEY, DATABASE_URL, REDIS_URL, SECRET_KEY
```

**2. Положите документ в `data/`**

```bash
cp your_regulations.txt data/document.txt
```

> OCR для PDF не поддерживается — используйте TXT или текстовые PDF.

**3. Запустите**

```bash
docker compose up --build
```

`entrypoint.sh` при первом запуске автоматически применит Alembic-миграции, построит FAISS-индекс и запустит сервер.

Swagger: `http://localhost:8000/docs`

**Ручная индексация**

```bash
python -m app.vectorstore.builder
# или через Docker:
docker compose run --rm app python -m app.vectorstore.builder
```

**Тесты**

```bash
pytest tests/ -v
```

---

## API

### Регистрация

```
POST /api/v1/auth/register
```

```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "api_key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

Ключ отображается **один раз**.

---

### Текстовый запрос

```
POST /api/v1/chat/text
X-API-Key: <ключ>
X-User-ID: <user_id>
Content-Type: multipart/form-data

message: "Сколько дней отпуска положено сотрудникам?"
image: <файл> (опционально)
```

### Аудио-запрос

```
POST /api/v1/chat/audio
X-API-Key: <ключ>
X-User-ID: <user_id>
Content-Type: multipart/form-data

audio: <mp3>
image: <файл> (опционально)
```

### Формат ответа

```json
{
  "content": "Ежегодный оплачиваемый отпуск составляет 28 календарных дней...",
  "used_chunk_indices": [0, 2],
  "source_chunks": [
    {
      "source": "data/document.txt",
      "content": "Ежегодный основной оплачиваемый отпуск: 28 календарных дней...",
      "confidence_score": 0.91
    }
  ]
}
```

`used_chunk_indices` — номера чанков, которые GPT-4o реально использовала при генерации ответа (из всех кандидатов, возвращённых FAISS). Позволяет отлаживать качество retrieval: если модель стабильно игнорирует часть чанков — стоит пересмотреть параметры поиска.

Для аудио-запроса дополнительно возвращается поле `transcription` с текстом от Whisper.

### Очистка памяти

```
DELETE /api/v1/chat/memory
X-API-Key: <ключ>
X-User-ID: <user_id>
```

---

## Тесты

```
tests/
├── test_rag.py          # Компоненты RAG-цепочки
├── test_repository.py   # CRUD через мок-сессию
└── test_security.py     # Генерация и верификация API-ключей
```

---

## Telegram-бот

Отдельный репозиторий на Aiogram 3, работает поверх этого API:
[https://github.com/HelixA350/telegram_bot](https://github.com/HelixA350/telegram_bot)
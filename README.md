# AI Consultant API

ИИ-консультант на основе документа. REST API на FastAPI с поддержкой текстовых, аудио и визуальных запросов.

## Стек

- Python 3.11+, FastAPI, LangChain, OpenAI API, FAISS, SQLAlchemy 2, PostgreSQL, Docker Compose

## Быстрый старт

### 1. Настройка переменных окружения

```bash
cp .env.example .env
# Отредактируйте .env: укажите OPENAI_API_KEY, SECRET_KEY
```

### 2. Добавьте документ

Положите ваш документ (PDF или TXT) в папку `data/`:

```
data/document.pdf
```

Или измените `DOCUMENT_PATH` в `.env`.

### 3. Запустите индексацию (первый раз)

Перед запуском сервиса необходимо проиндексировать документ:

```bash
# Локально (с установленными зависимостями)
pip install -r requirements.txt
python -m app.vectorstore.builder

# Или через Docker
docker compose run --rm app python -m app.vectorstore.builder
```

### 4. Запуск через Docker Compose

```bash
docker compose up --build
```

Сервис будет доступен на `http://localhost:8000`.

## API Endpoints

### Регистрация

```
POST /api/v1/auth/register
```

Возвращает API-ключ (показывается один раз).

### Текстовый запрос

```
POST /api/v1/chat/text
Headers: X-API-Key: <your-key>
Body: multipart/form-data
  - message: str (обязательно)
  - image: file (опционально, jpeg/png/webp/gif)
```

### Аудио-запрос

```
POST /api/v1/chat/audio
Headers: X-API-Key: <your-key>
Body: multipart/form-data
  - audio: file (обязательно, mp3)
  - image: file (опционально)
```

### Очистка памяти

```
DELETE /api/v1/chat/memory
Headers: X-API-Key: <your-key>
```

## Документация API

После запуска: `http://localhost:8000/docs`

## Структура проекта

```
project/
├── app/
│   ├── main.py              # точка входа FastAPI
│   ├── config.py            # настройки через Pydantic Settings
│   ├── dependencies.py      # зависимости FastAPI (auth, db)
│   ├── exceptions.py        # кастомные исключения
│   ├── schemas.py           # Pydantic схемы запросов/ответов
│   ├── api/v1/
│   │   ├── auth.py          # регистрация
│   │   └── chat.py          # чат-эндпоинты
│   ├── core/
│   │   ├── security.py      # генерация/проверка API-ключей
│   │   └── memory.py        # менеджер памяти диалогов
│   ├── services/
│   │   ├── rag.py           # RAG-цепочка
│   │   ├── transcription.py # Whisper транскрипция
│   │   └── vision.py        # GPT-4o Vision
│   ├── db/
│   │   ├── base.py          # SQLAlchemy движок и сессия
│   │   ├── models.py        # ORM-модели
│   │   └── repository.py    # CRUD операции
│   └── vectorstore/
│       ├── builder.py       # скрипт индексации
│       └── loader.py        # загрузка индекса
├── data/                    # документы для базы знаний
├── vectorstore/             # сохранённый FAISS-индекс
├── alembic/                 # миграции БД
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

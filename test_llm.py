"""
Тест: смотрим что именно уходит на neuroapi при разных методах structured output.
"""

import asyncio
import json
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import httpx
import logging

OPENAI_API_KEY = "sk-bu7EXgJ1euOJdWUMuTL4XQ"
OPENAI_BASE_URL = "https://litellm.tokengate.ru/v1"
MODEL = "openai/gpt-4.1"

MESSAGES = [
    SystemMessage(content="Ты помощник. Отвечай кратко."),
    HumanMessage(content="Как зовут первого президента США?"),
]


class Answer(BaseModel):
    content: str = Field(description="Ответ на вопрос.")
    confidence: int = Field(description="Уверенность от 0 до 100.")


llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY,
    verbose=True,
    base_url=OPENAI_BASE_URL,
    
).with_structured_output(Answer)

if __name__ == "__main__":
    print(llm.invoke(MESSAGES))
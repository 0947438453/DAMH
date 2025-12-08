# app/services/llm.py
from typing import List, Dict
import requests
from app.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


class LLMError(RuntimeError):
    pass


def chat_llm(messages: List[Dict[str, str]]) -> str:
    """
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Gọi Ollama /api/chat, có xử lý lỗi cơ bản.
    """
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
    }
    url = f"{OLLAMA_BASE_URL}/api/chat"

    try:
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Lỗi khi gọi LLM tại {url}: {e}")
    except KeyError:
        raise LLMError("Phản hồi từ LLM không đúng định dạng mong đợi.")

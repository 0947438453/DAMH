from typing import List, Dict
import requests
from app.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


def chat_llm(messages: List[Dict[str, str]]) -> str:
    """
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Gọi Ollama /api/chat
    """
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
    }
    url = f"{OLLAMA_BASE_URL}/api/chat"
    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    # Ollama /api/chat trả về "message": {"role": "...", "content": "..."}
    return data["message"]["content"].strip()

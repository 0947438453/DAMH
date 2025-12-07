# app/services/web_search.py

from typing import List
import requests
from app.config import TAVILY_API_KEY


def web_search(query: str, num_results: int = 3) -> List[str]:
    """
    Search web bằng Tavily, trả về list snippet text.
    """
    if not TAVILY_API_KEY:
        return [f"(Chưa cấu hình TAVILY_API_KEY trong .env, không thể search '{query}')"]

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": num_results,
        "search_depth": "basic",  # hoặc 'advanced'
    }

    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()

    snippets: List[str] = []
    for item in data.get("results", [])[:num_results]:
        title = item.get("title", "")
        content = item.get("content", "")
        link = item.get("url", "")
        snippets.append(f"{title} - {content} ({link})")

    if not snippets:
        snippets.append(f"(Không tìm thấy kết quả cho '{query}')")
    return snippets

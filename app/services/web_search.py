# app/services/web_search.py
from typing import List
import requests
from app.config import TAVILY_API_KEY


class WebSearchError(RuntimeError):
    pass


def web_search(query: str, num_results: int = 3) -> List[str]:
    """
    Search web bằng Tavily, trả về list snippet text.
    """
    if not TAVILY_API_KEY:
        # Không ném exception để /chat vẫn chạy được
        return [f"(Chưa cấu hình TAVILY_API_KEY trong .env, không thể search '{query}')"]

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": num_results,
    }

    try:
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        raise WebSearchError(f"Lỗi khi gọi Tavily: {e}")

    snippets: List[str] = []
    for item in data.get("results", [])[:num_results]:
        title = item.get("title", "")
        content = item.get("content", "")
        link = item.get("url", "")
        snippets.append(f"{title} - {content} ({link})")

    if not snippets:
        snippets.append(f"(Không tìm thấy kết quả cho '{query}')")
    return snippets

from typing import List, Tuple
from app.rag.vector_store import SimpleVectorStore
from app.services.embeddings import embed_text
from app.services.web_search import web_search
from app.services.llm import chat_llm


def classify_source(question: str) -> str:
    """
    Rule đơn giản: nếu có từ khoá thời gian/thời sự -> web, còn lại -> local.
    Bạn có thể nâng cấp thành dùng LLM sau.
    """
    q_lower = question.lower()
    web_keywords = [
        "hôm nay",
        "hiện tại",
        "mới nhất",
        "tin tức",
        "news",
        "thời tiết",
        "tỷ giá",
        "tỉ giá",
        "giá bao nhiêu",
    ]
    if any(k in q_lower for k in web_keywords):
        return "web"
    return "local"


def build_context_from_local(question: str, top_k: int = 5) -> str:
    vs = SimpleVectorStore(name="default")
    q_emb = embed_text(question)
    results: List[Tuple[str, float]] = vs.search(q_emb, top_k=top_k)
    if not results:
        return ""
    parts = []
    for text, dist in results:
        parts.append(f"- (dist={dist:.3f}) {text}")
    return "\n".join(parts)


def build_context_from_web(question: str, num_results: int = 3) -> str:
    results = web_search(question, num_results=num_results)
    return "\n".join([f"- {r}" for r in results])


def answer_question(question: str, source: str = "auto") -> tuple[str, List[str]]:
    used: List[str] = []

    if source == "auto":
        source = classify_source(question)

    local_ctx = ""
    web_ctx = ""

    if source in ["local", "both"]:
        local_ctx = build_context_from_local(question)
        if local_ctx:
            used.append("local")

    if source in ["web", "both"]:
        web_ctx = build_context_from_web(question)
        if web_ctx:
            used.append("web")

    context_blocks = []
    if local_ctx:
        context_blocks.append("DỮ LIỆU NỘI BỘ:\n" + local_ctx)
    if web_ctx:
        context_blocks.append("THÔNG TIN TỪ INTERNET:\n" + web_ctx)

    full_context = "\n\n".join(context_blocks) if context_blocks else "(Không có context)"

    system_prompt = """
Bạn là trợ lý tiếng Việt.
- Ưu tiên DỮ LIỆU NỘI BỘ nếu có mâu thuẫn với Internet.
- Nếu không thấy thông tin cần thiết trong context, hãy nói rõ: "Tôi không thấy thông tin trong dữ liệu hiện có."
- Trả lời ngắn gọn, rõ ràng.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{full_context}"},
        {"role": "user", "content": question},
    ]

    answer = chat_llm(messages)
    return answer, used

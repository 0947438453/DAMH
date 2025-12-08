from typing import List, Tuple
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.schemas import ChatRequest, ChatResponse
from app.services.llm import chat_llm, LLMError
from app.services.web_search import web_search, WebSearchError
from app.rag.vector_store import SimpleVectorStore


app = FastAPI(title="Chatbot học vụ")


# ====== CORS (nếu sau này bạn tách frontend riêng) ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== PHÂN LOẠI CÂU HỎI ======
def classify_question(question: str) -> str:
    """
    Phân loại câu hỏi vào 4 nhóm:
    - REGULATION: quy chế/điều khoản, xét tốt nghiệp, học vụ.
    - TUITION: học phí, lệ phí.
    - SCHEDULE: lịch học, thời khoá biểu của lớp/môn/tuần.
    - GENERAL: kiến thức chung / ngoài lề.
    """
    system_prompt = (
        "Bạn phân loại CÂU HỎI của người dùng vào một trong 4 nhóm sau "
        "(chỉ trả về đúng MỘT từ khoá, viết hoa):\n"
        "- REGULATION: câu hỏi về quy chế, điều khoản, quy định học vụ, xét tốt nghiệp...\n"
        "- TUITION: câu hỏi về học phí, lệ phí, chính sách thu chi.\n"
        "- SCHEDULE: câu hỏi về lịch học, thời khoá biểu, thời gian học của một lớp/môn/tuần cụ thể.\n"
        "- GENERAL: câu hỏi kiến thức chung hoặc ngoài lề (ví dụ AI, thời sự...).\n"
        "Bạn CHỈ trả về một trong 4 từ: REGULATION, TUITION, SCHEDULE hoặc GENERAL."
    )
    label = chat_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    )
    return label.strip().upper()

CLASS_CODE_RE = re.compile(r"\b\d{2}[A-Z]{2}\d{4}\b")


def extract_class_code(text: str) -> str | None:
    """
    Tìm mã lớp kiểu 25TH0101, 25AV0101...
    """
    m = CLASS_CODE_RE.search(text.upper())
    return m.group(0) if m else None


def extract_week(text: str) -> int | None:
    """
    Tìm 'tuần 15', 'tuan 15' trong câu hỏi.
    """
    m = re.search(r"tu[ầa]n\s*(\d+)", text.lower())
    return int(m.group(1)) if m else None


# ====== BUILD CONTEXT (LOCAL + WEB, tuỳ loại câu hỏi) ======
def build_context(question: str) -> Tuple[str, List[str]]:
    used_sources: List[str] = []
    context_blocks: List[str] = []

    MIN_LOCAL_SCORE = 0.20

    # 1) Phân loại câu hỏi + trích mã lớp / tuần
    try:
        label = classify_question(question)
    except Exception:
        label = "GENERAL"

    class_code = extract_class_code(question)
    week = extract_week(question)

    # ===== TRƯỜNG HỢP LỊCH HỌC (SCHEDULE) =====
    if label == "SCHEDULE":
        try:
            vs = SimpleVectorStore(name="default")
            # lấy nhiều chunk hơn một chút
            local_results = vs.search(question, top_k=20)  # List[(text, score)]

            schedule_filtered: List[Tuple[str, float]] = []
            for text, score in local_results:
                # phải có mã lớp
                if class_code and class_code in text:
                    # nếu người dùng hỏi kèm tuần thì lọc thêm theo tuần
                    if week is not None:
                        if f"tuần {week}" in text.lower() or f"tuan {week}" in text.lower():
                            schedule_filtered.append((text, score))
                    else:
                        schedule_filtered.append((text, score))

            if schedule_filtered:
                used_sources.append("local")
                for text, score in schedule_filtered:
                    context_blocks.append(f"[LOCAL schedule score={score:.2f}] {text}")
            else:
                # Không tìm được gì phù hợp -> ghi chú rõ cho LLM
                msg = (
                    f"[SYSTEM_NOTE] Không tìm thấy thông tin lịch học"
                    f"{f' tuần {week}' if week else ''}"
                    f"{f' của lớp {class_code}' if class_code else ''} "
                    f"trong các file lịch học đã nạp."
                )
                context_blocks.append(msg)

        except Exception as e:
            context_blocks.append(f"(Lỗi khi truy vấn dữ liệu local: {e})")

        # Với lịch học: không dùng web
        context = "\n\n".join(context_blocks)
        return context, used_sources

    # ===== CÁC LOẠI CÂU HỎI KHÁC GIỮ NGUYÊN NHƯ CŨ =====
    # Quyết định có dùng web không
    use_web = True
    if label in ("REGULATION", "CURRICULUM"):
        use_web = False

    # 2) Local RAG (giống code trước đây của bạn)
    try:
        vs = SimpleVectorStore(name="default")
        top_k = 5 if label != "GENERAL" else 3
        local_results = vs.search(question, top_k=top_k)
        filtered = [(t, s) for (t, s) in local_results if s >= MIN_LOCAL_SCORE]

        if filtered:
            used_sources.append("local")
            for text, score in filtered:
                context_blocks.append(f"[LOCAL score={score:.2f}] {text}")
    except Exception as e:
        context_blocks.append(f"(Lỗi khi truy vấn dữ liệu local: {e})")

    # 3) Web search
    if use_web:
        try:
            web_results = web_search(question, num_results=3)
            if web_results:
                used_sources.append("web")
                for snippet in web_results:
                    context_blocks.append(f"[WEB] {snippet}")
        except WebSearchError as e:
            context_blocks.append(f"(Lỗi web search: {e})")

    used_sources = list(dict.fromkeys(used_sources))
    context = "\n\n".join(context_blocks)
    return context, used_sources



# ====== GIAO DIỆN HTML (Chatbot học vụ) ======
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="utf-8"/>
        <title>Chatbot học vụ</title>
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #f5f5f7;
                color: #111827;
            }
            .app-container {
                max-width: 900px;
                margin: 0 auto;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            header {
                padding: 16px 24px;
                background: #111827;
                color: #f9fafb;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            header h1 {
                margin: 0;
                font-size: 20px;
            }
            header span.badge {
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 999px;
                border: 1px solid rgba(249,250,251,0.4);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            main {
                flex: 1;
                display: flex;
                flex-direction: column;
                padding: 16px 24px 8px;
            }
            .chat-window {
                flex: 1;
                border-radius: 16px;
                background: white;
                border: 1px solid #e5e7eb;
                padding: 12px;
                overflow-y: auto;
                max-height: calc(100vh - 210px);
            }
            .empty-state {
                text-align: center;
                padding: 32px 16px;
                color: #9ca3af;
                font-size: 14px;
            }
            .message {
                margin-bottom: 10px;
                display: flex;
                flex-direction: column;
                max-width: 80%;
            }
            .message.user {
                margin-left: auto;
                align-items: flex-end;
            }
            .message.bot {
                margin-right: auto;
                align-items: flex-start;
            }
            .bubble {
                padding: 10px 12px;
                border-radius: 16px;
                font-size: 14px;
                line-height: 1.4;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .message.user .bubble {
                background: #2563eb;
                color: white;
                border-bottom-right-radius: 4px;
            }
            .message.bot .bubble {
                background: #f3f4f6;
                color: #111827;
                border-bottom-left-radius: 4px;
            }
            .meta {
                font-size: 11px;
                color: #9ca3af;
                margin-top: 2px;
            }
            .meta span.source {
                background: #e5e7eb;
                border-radius: 999px;
                padding: 2px 8px;
            }
            .input-area {
                padding: 8px 24px 16px;
                border-top: 1px solid #e5e7eb;
                background: #f9fafb;
            }
            .input-inner {
                display: flex;
                gap: 8px;
                align-items: flex-end;
            }
            textarea {
                flex: 1;
                resize: none;
                min-height: 44px;
                max-height: 120px;
                padding: 10px 12px;
                border-radius: 12px;
                border: 1px solid #d1d5db;
                font-size: 14px;
                font-family: inherit;
            }
            textarea:focus {
                outline: none;
                border-color: #2563eb;
                box-shadow: 0 0 0 1px rgba(37,99,235,0.3);
            }
            button.send-btn {
                padding: 10px 16px;
                border-radius: 999px;
                border: none;
                background: #2563eb;
                color: white;
                font-size: 14px;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            button.send-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            button.send-btn:hover:not(:disabled) {
                background: #1d4ed8;
            }
            .spinner {
                width: 14px;
                height: 14px;
                border-radius: 999px;
                border: 2px solid rgba(255,255,255,0.6);
                border-top-color: white;
                animation: spin 0.7s linear infinite;
            }
            @keyframes spin { to { transform: rotate(360deg);} }
            footer {
                font-size: 11px;
                color: #9ca3af;
                padding: 0 24px 8px;
                text-align: right;
            }
            @media (max-width: 640px) {
                .app-container { max-width: 100%; }
                main { padding: 12px 12px 4px; }
                .input-area { padding: 8px 12px 12px; }
            }
        </style>
    </head>
    <body>
        <div class="app-container">
            <header>
                <div>
                    <h1>Chatbot học vụ</h1>
                    <div style="font-size:12px;opacity:0.8;">
                        Hỏi đáp về chương trình đào tạo, quy chế, học phí, lịch học...
                    </div>
                </div>
            </header>

            <main>
                <div id="chat" class="chat-window">
                    <div class="empty-state">
                        Hãy bắt đầu bằng cách nhập câu hỏi phía dưới.<br/>
                        Ví dụ:<br/>
                        – "Chương trình đào tạo ngành CNTT"<br/>
                        – "Quy định về xét tốt nghiệp"<br/>
                        – "Học phí học kỳ này là bao nhiêu?"<br/>
                    </div>
                </div>
            </main>

            <div class="input-area">
                <div class="input-inner">
                    <textarea id="question" placeholder="Nhập câu hỏi... (Enter để gửi, Shift+Enter để xuống dòng)"></textarea>
                    <button id="sendBtn" class="send-btn" onclick="sendMessage()">
                        <span id="sendLabel">Gửi</span>
                        <span id="sendSpinner" class="spinner" style="display:none;"></span>
                    </button>
                </div>
            </div>

            <footer>Demo học thuật.</footer>
        </div>

        <script>
            const chatEl = document.getElementById('chat');
            const questionEl = document.getElementById('question');
            const sendBtn = document.getElementById('sendBtn');
            const sendLabel = document.getElementById('sendLabel');
            const sendSpinner = document.getElementById('sendSpinner');

            function appendMessage(role, text, sources) {
                const empty = chatEl.querySelector('.empty-state');
                if (empty) empty.remove();

                const msg = document.createElement('div');
                msg.className = 'message ' + (role === 'user' ? 'user' : 'bot');

                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.textContent = text;
                msg.appendChild(bubble);

                if (role === 'bot' && sources && sources.length > 0) {
                    const meta = document.createElement('div');
                    meta.className = 'meta';
                    meta.innerHTML = '<span class="source">Nguồn: ' + sources.join(', ') + '</span>';
                    msg.appendChild(meta);
                }

                chatEl.appendChild(msg);
                chatEl.scrollTop = chatEl.scrollHeight;
            }

            function setLoading(isLoading) {
                sendBtn.disabled = isLoading;
                sendSpinner.style.display = isLoading ? 'inline-block' : 'none';
                sendLabel.textContent = isLoading ? 'Đang trả lời...' : 'Gửi';
            }

            async function sendMessage() {
                const question = questionEl.value.trim();
                if (!question || sendBtn.disabled) return;

                appendMessage('user', question);
                questionEl.value = '';
                setLoading(true);

                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });
                    const data = await res.json();

                    if (!res.ok) {
                        const detail = data.detail || 'Lỗi không xác định';
                        appendMessage('bot', '❌ Lỗi: ' + detail, []);
                    } else {
                        appendMessage('bot', data.answer, data.used_sources || []);
                    }
                } catch (err) {
                    appendMessage('bot', '❌ Lỗi kết nối tới server: ' + err, []);
                } finally {
                    setLoading(false);
                    questionEl.focus();
                }
            }

            questionEl.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ====== /chat ======
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    try:
        context, used_sources = build_context(req.question)

        system_prompt = (
            "Bạn là chatbot hỗ trợ học vụ của Trường Đại học Bình Dương. "
            "Trong mọi câu trả lời, bạn phải xưng là 'tôi'.\n\n"

            "Bạn phải PHÂN BIỆT 2 LOẠI CÂU HỎI:\n"
            "1) CÂU HỎI VỀ QUY CHẾ / ĐIỀU KIỆN HỌC VỤ (ví dụ: xét tốt nghiệp, xử lý vi phạm, học lại, bảo lưu...).\n"
            "   - Với LOẠI NÀY, bạn được phép trích dẫn Điều, Khoản trong Quy chế và NÊN nêu rõ nếu có.\n"
            "   - Phải liệt kê ĐẦY ĐỦ các điều kiện, trường hợp và ngoại lệ có trong ngữ cảnh, "
            "     không được bỏ sót điều kiện quan trọng.\n"
            "   - Nếu thực sự không tìm thấy điều khoản liên quan trong ngữ cảnh, hãy nói: "
            "     'Trong Quy chế đào tạo hiện tại tôi không thấy điều khoản rõ về vấn đề này, nên tôi không chắc.'\n\n"

            "2) CÂU HỎI KIẾN THỨC CHUNG / GIỚI THIỆU / KHÔNG LIÊN QUAN TRỰC TIẾP ĐẾN QUY CHẾ\n"
            "   (ví dụ: ChatGPT là gì, giới thiệu về trường, hỏi về AI, tin tức,...).\n"
            "   - Với LOẠI NÀY, TUYỆT ĐỐI KHÔNG được nhắc tới 'Điều', 'Khoản', 'Quy chế', "
            "     cũng KHÔNG nói các câu như 'tôi không thấy Điều, Khoản nào...'.\n"
            "   - Chỉ tập trung giải thích nội dung câu hỏi dựa trên ngữ cảnh local và web.\n\n"

            "Quy tắc dùng LOCAL & WEB:\n"
            "- Ưu tiên thông tin LOCAL khi câu hỏi liên quan đến trường, quy chế, chương trình đào tạo, học phí.\n"
            "- Nếu dùng WEB, hãy nói rõ 'theo thông tin tham khảo từ web' rồi tổng hợp nội dung đầy đủ, "
            "  không chỉ chép lại một câu ngắn.\n\n"

            "Cách trình bày câu trả lời:\n"
            "- Luôn trả lời HOÀN TOÀN bằng TIẾNG VIỆT.\n"
            "- Trả lời súc tích nhưng ĐẦY ĐỦ ý chính (điều kiện, mốc thời gian, ngoại lệ, ví dụ nếu có).\n"
            "- Ưu tiên dùng gạch đầu dòng hoặc đánh số để người đọc dễ theo dõi.\n"
            "- Không lặp lại nguyên văn ngữ cảnh; hãy diễn đạt lại cho dễ hiểu.\n\n"

            "Nếu trong NGỮ CẢNH có dòng bắt đầu bằng [SYSTEM_NOTE] thì bạn "
            "PHẢI làm đúng theo nội dung dòng đó và KHÔNG được suy đoán hay bịa thêm thông tin."
        )

        if context:
            user_content = (
                f"Người dùng hỏi: {req.question}\n\n"
                f"Ngữ cảnh (từ tài liệu & web):\n{context}\n\n"
                "Hãy TRẢ LỜI BẰNG TIẾNG VIỆT, rõ ràng, có thể đánh số/gạch đầu dòng. "
                "Nếu trong ngữ cảnh có Điều, Khoản liên quan thì hãy nêu rõ."
            )
        else:
            user_content = (
                f"Người dùng hỏi: {req.question}\n\n"
                "Hiện tại không có ngữ cảnh từ tài liệu hoặc web. "
                "Hãy trả lời chung chung nếu có thể, hoặc nói rõ là bạn không chắc."
            )

        answer = chat_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )

        return ChatResponse(answer=answer, used_sources=used_sources)

    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Lỗi khi gọi mô hình LLM: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ server: {e}")


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ====== Chạy trực tiếp: python -m app.main ======
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=9000,
        reload=True,
    )

# app/main.py  — bản test giao diện đơn giản

from typing import List
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Test Chat UI với FastAPI")


# ==== Models đơn giản cho /chat ====
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    used_sources: List[str]


# ==== Giao diện chat (GET /) ====
@app.get("/", response_class=HTMLResponse)
def index():
    # HTML cực đơn giản, chỉ để chắc chắn route "/" chạy
    html = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8" />
        <title>Test Chat UI</title>
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f4f4f4;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .chat-container {
                background: #ffffff;
                width: 420px;
                max-width: 100%;
                height: 90vh;
                max-height: 650px;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-header {
                padding: 12px 16px;
                background: #2563eb;
                color: #fff;
                font-weight: bold;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .chat-messages {
                flex: 1;
                padding: 12px;
                overflow-y: auto;
                background: #f9fafb;
            }
            .msg {
                margin-bottom: 10px;
                display: flex;
            }
            .msg.user { justify-content: flex-end; }
            .msg.assistant { justify-content: flex-start; }
            .bubble {
                max-width: 80%;
                padding: 8px 12px;
                border-radius: 12px;
                font-size: 14px;
                line-height: 1.4;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .msg.user .bubble {
                background: #2563eb;
                color: #fff;
                border-bottom-right-radius: 2px;
            }
            .msg.assistant .bubble {
                background: #e5e7eb;
                color: #111827;
                border-bottom-left-radius: 2px;
            }
            .chat-input {
                padding: 8px;
                border-top: 1px solid #e5e7eb;
                display: flex;
                gap: 6px;
            }
            .chat-input input {
                flex: 1;
                padding: 8px 10px;
                border-radius: 999px;
                border: 1px solid #d1d5db;
                outline: none;
                font-size: 14px;
            }
            .chat-input button {
                padding: 8px 14px;
                border-radius: 999px;
                border: none;
                background: #2563eb;
                color: #fff;
                font-size: 14px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <div>Test Chat UI</div>
            </div>
            <div id="messages" class="chat-messages">
                <div class="msg assistant">
                    <div class="bubble">
                        Xin chào 👋 Đây là bản test giao diện.
                        Hãy gõ câu hỏi và nhấn Gửi, bot sẽ lặp lại câu hỏi.
                    </div>
                </div>
            </div>
            <div class="chat-input">
                <input id="input" type="text" placeholder="Nhập câu hỏi..." />
                <button id="sendBtn">Gửi</button>
            </div>
        </div>

        <script>
            const inputEl = document.getElementById("input");
            const messagesEl = document.getElementById("messages");
            const sendBtn = document.getElementById("sendBtn");

            function appendMessage(role, text) {
                const msgDiv = document.createElement("div");
                msgDiv.className = "msg " + role;

                const bubble = document.createElement("div");
                bubble.className = "bubble";
                bubble.textContent = text;

                msgDiv.appendChild(bubble);
                messagesEl.appendChild(msgDiv);
                messagesEl.scrollTop = messagesEl.scrollHeight;
            }

            async function sendMessage() {
                const text = inputEl.value.trim();
                if (!text) return;

                appendMessage("user", text);
                inputEl.value = "";
                inputEl.focus();
                sendBtn.disabled = true;

                try {
                    const resp = await fetch("/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question: text })
                    });

                    if (!resp.ok) {
                        const errText = await resp.text();
                        appendMessage("assistant", "Lỗi server: " + errText);
                    } else {
                        const data = await resp.json();
                        appendMessage("assistant", data.answer);
                    }
                } catch (err) {
                    appendMessage("assistant", "Không gọi được API: " + err);
                } finally {
                    sendBtn.disabled = false;
                }
            }

            sendBtn.addEventListener("click", sendMessage);
            inputEl.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ==== API tối giản: chỉ echo lại câu hỏi ====
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return ChatResponse(
        answer=f"Bạn vừa hỏi: {req.question}",
        used_sources=[],
    )

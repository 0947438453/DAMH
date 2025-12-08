# app/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# Thư mục gốc project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load file .env ở root
load_dotenv(BASE_DIR / ".env")

# Thư mục dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Đảm bảo tồn tại
RAW_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình LLM (Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")

# Tavily API key (web search)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

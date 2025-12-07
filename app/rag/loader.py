# app/rag/loader.py

from pathlib import Path
from typing import List
import fitz       # PyMuPDF
import docx
import pandas as pd


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)


def load_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join([para.text for para in document.paragraphs])


def load_csv(path: Path) -> str:
    """
    Đọc file CSV và convert thành text.
    Bạn có thể tuỳ biến format (vd chỉ lấy 1 số cột).
    """
    df = pd.read_csv(path)
    return df.to_csv(index=False)  # hoặc df.to_string()


def load_excel(path: Path) -> str:
    """
    Đọc Excel (.xlsx, .xls) và convert thành text.
    - Gộp tất cả sheet lại
    """
    # đọc tất cả sheet
    xls = pd.ExcelFile(path)
    texts = []
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        # bạn có thể custom: chỉ lấy 1 số cột, rename cột, v.v.
        sheet_text = f"=== Sheet: {sheet_name} ===\n" + df.to_csv(index=False)
        texts.append(sheet_text)
    return "\n\n".join(texts)


def load_any(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return load_txt(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext in [".docx", ".doc"]:
        return load_docx(path)
    if ext == ".csv":
        return load_csv(path)
    if ext in [".xlsx", ".xls"]:
        return load_excel(path)
    raise ValueError(f"Không hỗ trợ định dạng file: {ext}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Chia text thành nhiều chunk (tính theo số từ) để dùng RAG.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

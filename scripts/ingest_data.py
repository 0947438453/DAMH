from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from app.config import RAW_DIR
from app.rag.loader import load_any, chunk_text
from app.rag.vector_store import SimpleVectorStore
from app.services.embeddings import embed_texts


# Cấu hình metadata cho từng file (theo tên file trong RAW_DIR)
FILE_CONFIG: Dict[str, Dict[str, Any]] = {
    # ví dụ: đặt file PDF vào RAW_DIR với đúng tên dưới đây
    "Quy_che_đao_tao.pdf": {
        "doc_id": "quy_che_dao_tao",
        "doc_type": "regulation",  # quy chế
        "title": "Quy chế đào tạo",
    },
    "Thong bao thu HP HK1 2025-2026.pdf": {
        "doc_id": "hoc_phi_hk1_2025_2026",
        "doc_type": "tuition",  # học phí
        "title": "Thông tin học phí hk1 2025-2026",
    },
    "Tuan_15.pdf": {
        "doc_id": "lich_hoc_tuan_15_2025",
        "doc_type": "schedule",  # lịch học
        "title": "Lịch học tuần 15 (08–14/12/2025)",
    },
    # thêm các file khác ở đây...
}


def infer_metadata(path: Path) -> Dict[str, Any]:
    """
    Nếu file không nằm trong FILE_CONFIG thì suy ra metadata cơ bản.
    """
    name = path.name
    stem = path.stem
    lower = stem.lower()

    # đoán sơ loại tài liệu theo tên file
    if "quy_che" in lower or "quy che" in lower:
        doc_type = "regulation"
    elif "hoc_phi" in lower or "hoc phi" in lower:
        doc_type = "tuition"
    elif "lich_hoc" in lower or "lich hoc" in lower:
        doc_type = "schedule"
    else:
        doc_type = "general"

    return {
        "doc_id": stem,
        "doc_type": doc_type,
        "title": name,
    }


def ingest_folder(folder: Path, store_name: str = "default"):
    """
    Đọc tất cả file trong RAW_DIR, chunk text, tạo embedding và lưu vào vector store.
    Mỗi chunk đều kèm metadata (doc_id, doc_type, title).
    """
    vs = SimpleVectorStore(name=store_name)

    files = [p for p in folder.glob("**/*") if p.is_file()]
    all_chunks: List[tuple[str, Dict[str, Any]]] = []

    for path in tqdm(files, desc="Đọc file"):
        try:
            text = load_any(path)
        except ValueError:
            print(f"Bỏ qua (không hỗ trợ định dạng): {path}")
            continue

        chunks = chunk_text(text)

        # lấy metadata từ FILE_CONFIG nếu có, ngược lại suy ra tự động
        meta = FILE_CONFIG.get(path.name)
        if meta is None:
            meta = infer_metadata(path)

        for ch in chunks:
            all_chunks.append((ch, meta))

    print(f"Tổng số chunks: {len(all_chunks)}")

    batch_size = 32
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Tạo embedding"):
        batch = all_chunks[i:i + batch_size]
        texts = [t for (t, _) in batch]
        metas = [m for (_, m) in batch]

        emb = embed_texts(texts)
        # YÊU CẦU: SimpleVectorStore.add phải nhận được metadatas
        vs.add(emb, texts)

    print("Hoàn tất ingest.")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ingest_folder(RAW_DIR, store_name="default")

from pathlib import Path
from tqdm import tqdm

from app.config import RAW_DIR
from app.rag.loader import load_any, chunk_text
from app.rag.vector_store import SimpleVectorStore
from app.services.embeddings import embed_texts


def ingest_folder(folder: Path, store_name: str = "default"):
    vs = SimpleVectorStore(name=store_name)

    files = [p for p in folder.glob("**/*") if p.is_file()]
    all_chunks = []

    for path in tqdm(files, desc="Đọc file"):
        try:
            text = load_any(path)
        except ValueError:
            print(f"Bỏ qua: {path}")
            continue
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    print(f"Tổng số chunks: {len(all_chunks)}")

    batch_size = 32
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Tạo embedding"):
        batch = all_chunks[i:i + batch_size]
        emb = embed_texts(batch)
        vs.add(emb, batch)

    print("Hoàn tất ingest.")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ingest_folder(RAW_DIR, store_name="default")

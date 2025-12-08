# app/rag/vector_store.py
from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import numpy as np
import pickle

from app.config import VECTOR_STORE_DIR
from app.services.embeddings import embed_texts


class SimpleVectorStore:
    """
    Lưu text + embedding ra đĩa, cho phép search theo cosine similarity.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        self.emb_path = VECTOR_STORE_DIR / f"{name}_embeddings.npy"
        self.texts_path = VECTOR_STORE_DIR / f"{name}_texts.pkl"

        if self.emb_path.exists() and self.texts_path.exists():
            self.embeddings = np.load(self.emb_path)
            with open(self.texts_path, "rb") as f:
                self.texts: List[str] = pickle.load(f)
        else:
            self.embeddings = np.empty((0, 512), dtype="float32")
            self.texts: List[str] = []

    def _save(self) -> None:
        np.save(self.emb_path, self.embeddings)
        with open(self.texts_path, "wb") as f:
            pickle.dump(self.texts, f)

    def add(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """
        Thêm batch embedding + text.
        embeddings.shape = (batch_size, dim)
        """
        if embeddings.size == 0:
            return

        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.texts.extend(texts)
        self._save()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Tìm top_k đoạn text phù hợp với query, trả về [(text, score), ...]
        """
        if len(self.texts) == 0:
            return []

        query_emb = embed_texts([query])[0]  # (dim,)

        # cosine similarity
        emb = self.embeddings
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        sims = emb_norm @ q_norm  # (n,)
        idx = np.argsort(-sims)[:top_k]

        results: List[Tuple[str, float]] = []
        for i in idx:
            results.append((self.texts[i], float(sims[i])))
        return results

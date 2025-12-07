from pathlib import Path
from typing import List, Tuple
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from app.config import VECTOR_STORE_DIR


class SimpleVectorStore:
    def __init__(self, name: str = "default"):
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.path: Path = VECTOR_STORE_DIR / f"{name}.pkl"
        self.embeddings: np.ndarray | None = None
        self.texts: List[str] = []
        self.nn: NearestNeighbors | None = None

        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path, "rb") as f:
            obj = pickle.load(f)
        self.embeddings = obj["embeddings"]
        self.texts = obj["texts"]
        if self.embeddings is not None and len(self.embeddings) > 0:
            self.nn = NearestNeighbors(
                n_neighbors=min(5, len(self.embeddings)),
                metric="cosine",
            )
            self.nn.fit(self.embeddings)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(
                {"embeddings": self.embeddings, "texts": self.texts},
                f,
            )

    def add(self, embeddings: np.ndarray, texts: List[str]):
        if self.embeddings is None:
            self.embeddings = embeddings
            self.texts = texts
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.texts.extend(texts)

        self.nn = NearestNeighbors(
            n_neighbors=min(5, len(self.embeddings)),
            metric="cosine",
        )
        self.nn.fit(self.embeddings)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Trả về list (text, distance) – distance càng nhỏ càng giống
        """
        if self.nn is None or self.embeddings is None or len(self.embeddings) == 0:
            return []
        distances, indices = self.nn.kneighbors(
            query_embedding.reshape(1, -1),
            n_neighbors=min(top_k, len(self.embeddings)),
        )
        res: List[Tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            res.append((self.texts[idx], float(dist)))
        return res

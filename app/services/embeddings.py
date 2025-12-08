# app/services/embeddings.py
from typing import List
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Vectorizer đơn giản, 512 chiều
_vectorizer = HashingVectorizer(
    n_features=512,
    alternate_sign=False,
    norm=None
)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Nhận list string, trả về np.ndarray (n_samples, dim)
    """
    if not texts:
        return np.empty((0, 512), dtype="float32")
    X = _vectorizer.transform(texts)
    return X.toarray().astype("float32")


def embed_text(text: str) -> np.ndarray:
    """
    Embedding cho 1 câu
    """
    return embed_texts([text])[0]

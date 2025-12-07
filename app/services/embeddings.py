# app/services/embeddings.py

from typing import List
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Dùng HashingVectorizer để tạo vector văn bản (nhẹ, không phụ thuộc deep learning)
# n_features: số chiều vector, bạn có thể tăng lên nếu dữ liệu lớn hơn (1024, 2048,...)
_vectorizer = HashingVectorizer(
    n_features=512,        # số chiều vector
    alternate_sign=False,  # tránh giá trị âm
    norm="l2"              # chuẩn hóa, phù hợp cho cosine similarity
)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Tạo embedding cho nhiều đoạn text.
    Trả về numpy array shape (n, d) kiểu float32.
    """
    # HashingVectorizer trả dạng sparse matrix, convert sang dense
    X = _vectorizer.transform(texts)
    return X.toarray().astype("float32")


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]

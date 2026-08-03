"""Text embeddings for catalog retrieval.

`Embedder` is a small interface with two implementations:

* `HashingEmbedder` — the default. A hashed TF-IDF over word unigrams, word
  bigrams and character 4-grams, L2-normalized. Runs offline, needs no API key,
  and is fully deterministic, which keeps the committed index reproducible.
* `GeminiEmbedder` — real semantic embeddings when a key is available.

Being straight about the trade-off: hashed TF-IDF is *lexical*, not semantic.
It will match "oak dining table" to "oak table" but not to "wooden eating
surface". That is an acceptable default here because the heavy lifting in this
system is done by **hard structured filters** — category, dimensions, price,
colour and style are typed fields, not vibes, and they run in SQL before
anything is scored. The embedding only reranks an already-valid candidate set.
Point `EMBEDDING_BACKEND=gemini` at it when semantic recall actually matters.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Word unigrams, word bigrams, and char 4-grams of each word.

    The char n-grams give some robustness to plurals and compound words
    ("bookcase"/"bookcases", "armchair"/"chair") that pure unigrams miss.
    """
    words = _TOKEN.findall(text.lower())
    tokens: list[str] = list(words)
    tokens += [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
    for word in words:
        if len(word) > 5:
            tokens += [f"#{word[i : i + 4]}" for i in range(len(word) - 3)]
    return tokens


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized row vectors."""
        ...


class HashingEmbedder:
    """Deterministic hashed TF-IDF. No network, no model download."""

    def __init__(self, dim: int = 256, idf: np.ndarray | None = None) -> None:
        self.dim = dim
        self.idf = idf if idf is not None else np.ones(dim, dtype=np.float32)

    def fit(self, texts: list[str]) -> HashingEmbedder:
        """Learn bucket IDF weights over the corpus.

        Without this, common words like "table" dominate every vector and
        everything looks similar to everything.
        """
        n_docs = max(len(texts), 1)
        doc_freq = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            seen = {_bucket(tok, self.dim) for tok in _tokenize(text)}
            for bucket in seen:
                doc_freq[bucket] += 1.0
        self.idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)).astype(np.float32) + 1.0
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in _tokenize(text):
                bucket = _bucket(token, self.dim)
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
            for bucket, count in counts.items():
                # Sublinear TF: a word appearing 10× isn't 10× as relevant.
                matrix[row, bucket] = (1.0 + math.log(count)) * self.idf[bucket]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)


class GeminiEmbedder:
    """Real semantic embeddings. Used when EMBEDDING_BACKEND=gemini."""

    def __init__(self, api_key: str, model: str = "text-embedding-004", dim: int = 768) -> None:
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        # Batched — one request per product would be needlessly slow and costly.
        for start in range(0, len(texts), 100):
            response = self.client.models.embed_content(
                model=self.model, contents=texts[start : start + 100]
            )
            vectors.extend(e.values for e in response.embeddings)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)


def build_embedder(backend: str, api_key: str | None, dim: int) -> Embedder:
    if backend == "gemini" and api_key:
        return GeminiEmbedder(api_key=api_key)
    return HashingEmbedder(dim=dim)

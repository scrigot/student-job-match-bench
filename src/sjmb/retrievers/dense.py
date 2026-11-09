"""Dense retrieval with exact cosine search.

Encoders:
- `st`: sentence-transformers, local (`uv sync --extra dense`). Default model all-MiniLM-L6-v2.
- `azure`: an Azure AI Foundry embeddings deployment (AZURE_EMBED_DEPLOYMENT).
- `lsa`: TF-IDF + truncated SVD in numpy. No downloads; the offline fallback for CI and tests.
`auto` picks `st` when sentence-transformers is installed, otherwise `lsa`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from sjmb import azure
from sjmb.retrievers.base import tokenize
from sjmb.types import Posting


class Encoder(Protocol):
    name: str

    def fit(self, corpus: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class LSAEncoder:
    """TF-IDF + SVD. A weak dense baseline, but deterministic and dependency-free."""

    def __init__(self, dim: int = 128, vocab_size: int = 20000):
        self.dim, self.vocab_size = dim, vocab_size
        self.name = f"lsa-{dim}"

    def _tfidf(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for term, tf in Counter(tokenize(t)).items():
                j = self.vocab.get(term)
                if j is not None:
                    m[i, j] = (1 + np.log(tf)) * self.idf[j]
        return _normalize(m)

    def fit(self, corpus: list[str]) -> None:
        df: Counter[str] = Counter()
        for t in corpus:
            df.update(set(tokenize(t)))
        terms = [t for t, f in df.most_common(self.vocab_size) if f >= 2] or list(df)
        self.vocab = {t: i for i, t in enumerate(terms)}
        n = len(corpus)
        self.idf = np.array([np.log((1 + n) / (1 + df[t])) + 1 for t in terms], dtype=np.float32)
        x = self._tfidf(corpus)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        self.components = vt[: min(self.dim, vt.shape[0])].T

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(self._tfidf(texts) @ self.components)


class SentenceTransformerEncoder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.name = f"st-{model.split('/')[-1]}"

    def fit(self, corpus: list[str]) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=64, normalize_embeddings=True))


class AzureEmbeddingEncoder:
    """An Azure AI Foundry embeddings deployment through the OpenAI v1 route.

    Vectors are cached in SQLite keyed by a hash of (deployment, text), so the corpus is only
    embedded once per deployment.
    """

    def __init__(self, batch: int = 64, cache_path: Path | None = None, llm=None):
        self.deployment = os.environ["AZURE_EMBED_DEPLOYMENT"]
        self.batch = batch
        self.name = f"azure-{self.deployment}"
        self.db = sqlite3.connect(cache_path or azure.cache_root() / "embeddings.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self._llm = llm
        self.calls = 0

    @property
    def llm(self):
        if self._llm is None:
            self._llm = azure.client()
        return self._llm

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.deployment}\x00{text}".encode()).hexdigest()

    def uncached(self, texts: list[str]) -> list[str]:
        keys = {self._key(t[:8000]): t[:8000] for t in texts}
        have = set()
        items = list(keys)
        for i in range(0, len(items), 500):
            chunk = items[i : i + 500]
            q = f"SELECT key FROM emb WHERE key IN ({','.join('?' * len(chunk))})"
            have.update(k for (k,) in self.db.execute(q, chunk))
        return [t for k, t in keys.items() if k not in have]

    def estimate(self, texts: list[str]) -> tuple[int, float]:
        """(API calls, USD) to embed the uncached part of `texts`."""
        todo = self.uncached(texts)
        tokens = sum(azure.estimate_tokens(t) for t in todo)
        return -(-len(todo) // self.batch), azure.estimate_cost(self.deployment, tokens)

    def fit(self, corpus: list[str]) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [t[:8000] for t in texts]
        todo = self.uncached(texts)
        for i in range(0, len(todo), self.batch):
            chunk = todo[i : i + self.batch]
            resp = self.llm.embeddings.create(model=self.deployment, input=chunk)
            self.calls += 1
            rows = [
                (self._key(t), np.asarray(d.embedding, dtype=np.float32).tobytes())
                for t, d in zip(chunk, sorted(resp.data, key=lambda d: d.index), strict=True)
            ]
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self.db.commit()
        vecs = []
        for t in texts:
            (blob,) = self.db.execute(
                "SELECT vec FROM emb WHERE key = ?", (self._key(t),)
            ).fetchone()
            vecs.append(np.frombuffer(blob, dtype=np.float32))
        return _normalize(np.vstack(vecs))


def make_encoder(kind: str = "auto") -> Encoder:
    if kind == "auto":
        kind = "st" if importlib.util.find_spec("sentence_transformers") else "lsa"
    if kind == "st":
        return SentenceTransformerEncoder()
    if kind == "azure":
        return AzureEmbeddingEncoder()
    if kind == "lsa":
        return LSAEncoder()
    raise ValueError(f"unknown encoder {kind!r}")


class Dense:
    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.name = f"dense[{encoder.name}]"

    def fit(self, postings: list[Posting]) -> Dense:
        corpus = [p.text() for p in postings]
        self.encoder.fit(corpus)
        self.matrix = self.encoder.encode(corpus)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.encoder.encode([query])[0]
        return self.matrix @ q

    def scores_many(self, queries: list[str]) -> np.ndarray:
        return self.encoder.encode(queries) @ self.matrix.T

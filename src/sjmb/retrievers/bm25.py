"""Okapi BM25 over title + company + location + description."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from sjmb.retrievers.base import tokenize
from sjmb.types import Posting


class BM25:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, title_boost: int = 2):
        self.k1, self.b, self.title_boost = k1, b, title_boost

    def fit(self, postings: list[Posting]) -> BM25:
        docs = []
        for p in postings:
            # Repeat the title so a title hit outweighs a boilerplate hit in the description.
            docs.append(tokenize(" ".join([p.title] * (self.title_boost - 1)) + " " + p.text()))
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        df: Counter[str] = Counter()
        self.postings_tf: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                df[term] += 1
                self.postings_tf.setdefault(term, []).append((i, tf))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings_tf[term]:
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm[i])
        return out

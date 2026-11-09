"""Ranking metrics over graded (0-4) relevance.

Judgments come from a pooled judge run: unjudged documents count as grade 0, and the ideal
ranking for NDCG and the relevant set for recall are built from the judged pool. Both are the
standard TREC pooling assumptions, and both mean scores are relative to the pool depth used.
"""

from __future__ import annotations

import math

from sjmb.judge.rubric import POOR_MATCH_MAX, RELEVANT_MIN


def dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    gains = [qrels.get(d, 0) for d in ranked[:k]]
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    return dcg(gains) / ideal if ideal > 0 else 0.0


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int = 50) -> float:
    relevant = {d for d, g in qrels.items() if g >= RELEVANT_MIN}
    if not relevant:
        return math.nan  # undefined; excluded from the mean
    return len(relevant & set(ranked[:k])) / len(relevant)


def poor_match_rate_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """Over the results actually returned. An empty list is undefined (NaN), not a perfect 0."""
    top = ranked[:k]
    if not top:
        return math.nan
    return sum(1 for d in top if qrels.get(d, 0) <= POOR_MATCH_MAX) / len(top)


def violation_rate_at_k(ranked: list[str], ineligible: set[str], k: int = 10) -> float:
    top = ranked[:k]
    if not top:
        return math.nan
    return sum(1 for d in top if d in ineligible) / len(top)

"""LambdaMART reranker (KIT-254): learn the judge's ranking from cheap features.

The judge is an LLM call per pair. This trains a LightGBM LambdaMART model on the judge's grades
so that, at query time, ranking needs no LLM call at all. It reranks the eligible hybrid top-50.

Evaluation is cross-validated by profile: a student's pairs are never in both the training and
the test fold. Training uses only pairs the judge has already graded. At test time the model can
pull up a candidate nobody has judged, so its new top-10 is judged before scoring (pool
expansion), and the baseline is scored against the same, larger set of judgments.

The labels come from the judge, so this measures how well a cheap model reproduces the judge,
not agreement with people.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np

from sjmb.data.extract import restriction
from sjmb.eval.metrics import ndcg_at_k, paired_bootstrap
from sjmb.eval.runner import preflight
from sjmb.filters.eligibility import eligibility_mask
from sjmb.judge.judges import ADJACENT, Judge, location_match, same_track
from sjmb.retrievers import BM25, Dense, Hybrid, make_encoder, top_k
from sjmb.retrievers.base import tokenize
from sjmb.types import Posting, Profile

CANDIDATES = 50
FEATURES = (
    "bm25",
    "dense",
    "hybrid_rrf",
    "hybrid_rank",
    "track_same",
    "track_adjacent",
    "track_unrelated",
    "track_unknown",
    "location_match",
    "location_unknown",
    "level_match",
    "grad_window_stated",
    "sponsorship_stated",
    "restricted_program",
    "skill_hits",
    "interest_overlap",
    "log_desc_len",
)


def features(
    prof: Profile, post: Posting, bm25: float, dense: float, rrf: float, rank: int
) -> list[float]:
    track_known = post.track != "other"
    same = track_known and same_track(post.track, prof.track)
    adjacent = track_known and not same and post.track in ADJACENT[prof.track]
    loc = location_match(prof, post)
    text = set(tokenize(post.text()))
    skills = [set(tokenize(s)) for s in prof.skills]
    interests = set(tokenize(prof.interests))
    return [
        bm25,
        dense,
        rrf,
        float(rank),
        float(same),
        float(adjacent),
        float(track_known and not same and not adjacent),
        float(not track_known),
        float(loc is True),
        float(loc is None),
        float(post.role_type == prof.seeking),
        float(bool(post.grad_years)),
        float(post.sponsorship != "unknown"),
        float(restriction(post.title, post.description) is not None),
        float(sum(1 for s in skills if s and s <= text)),
        len(interests & text) / max(1, len(interests)),
        math.log1p(len(post.description)),
    ]


@dataclass
class LtrResult:
    profiles: int
    folds: int
    baseline: list[float]  # NDCG@10 per profile, hybrid-rrf + eligibility
    rules: list[float]  # the same, with the restricted-program check also applied
    ltr: list[float]  # NDCG@10 per profile, LambdaMART rerank
    new_judgments: int
    importance: dict[str, float]

    def to_markdown(self) -> str:
        mean, lo, hi = paired_bootstrap(self.rules, self.ltr)
        rows = sorted(self.importance.items(), key=lambda kv: -kv[1])
        total = sum(self.importance.values()) or 1.0
        return "\n".join(
            [
                "# LambdaMART rerank",
                "",
                f"{self.profiles} profiles, {self.folds}-fold cross-validation by profile, "
                f"reranking the eligible hybrid top-{CANDIDATES}. "
                f"{self.new_judgments} new judge calls to grade the new top-10s.",
                "",
                "| system | ndcg@10 |",
                "|---|---:|",
                f"| hybrid-rrf + eligibility | {np.mean(self.baseline):.3f} |",
                f"| + restricted-program check | {np.mean(self.rules):.3f} |",
                f"| + LambdaMART rerank | {np.mean(self.ltr):.3f} |",
                "",
                f"LambdaMART vs the rule baseline: {mean:+.3f}, "
                f"95% CI [{lo:+.3f}, {hi:+.3f}] (paired bootstrap over profiles).",
                "",
                "Feature importance (share of total gain):",
                "",
                "| feature | gain |",
                "|---|---:|",
                *[f"| {k} | {v / total:.1%} |" for k, v in rows],
            ]
        )


def run_ltr(
    postings: list[Posting],
    profiles: list[Profile],
    judge: Judge,
    as_of: date,
    encoder: str = "auto",
    folds: int = 5,
    seed: int = 0,
    progress: Callable[[str], None] = lambda _: None,
) -> LtrResult:
    import lightgbm as lgb

    bm25 = BM25().fit(postings)
    dense = Dense(make_encoder(encoder)).fit(postings)
    hybrid = Hybrid([bm25, dense]).fit(postings)

    cands: dict[str, list[int]] = {}
    feats: dict[str, np.ndarray] = {}
    for prof in profiles:
        q = prof.query()
        sb, sd, sh = bm25.scores(q), dense.scores(q), hybrid.scores(q)
        idx = list(top_k(sh, CANDIDATES, eligibility_mask(prof, postings, as_of)))
        cands[prof.id] = idx
        feats[prof.id] = np.array(
            [features(prof, postings[i], sb[i], sd[i], sh[i], r) for r, i in enumerate(idx, 1)]
        )

    # Labels: whatever the judge has already graded (no new calls for training).
    def cached_grade(prof: Profile, i: int) -> int | None:
        if hasattr(judge, "is_cached") and not judge.is_cached(prof, postings[i]):
            return None
        return judge.grade(prof, postings[i]).grade

    labels = {p.id: [cached_grade(p, i) for i in cands[p.id]] for p in profiles}

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(profiles))
    fold_of = {profiles[k].id: n % folds for n, k in enumerate(order)}
    picks: dict[str, list[int]] = {}
    gain = np.zeros(len(FEATURES))
    for f in range(folds):
        train = [p for p in profiles if fold_of[p.id] != f]
        X, y, group = [], [], []
        for p in train:
            keep = [k for k, g in enumerate(labels[p.id]) if g is not None]
            if len(keep) < 2:
                continue
            X.append(feats[p.id][keep])
            y += [labels[p.id][k] for k in keep]
            group.append(len(keep))
        params = {
            "objective": "lambdarank",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 10,
            "seed": seed,
            "verbose": -1,
        }
        data = lgb.Dataset(np.vstack(X), label=np.array(y), group=group)
        model = lgb.train(params, data, num_boost_round=200)
        gain += model.feature_importance("gain")
        for p in profiles:
            if fold_of[p.id] == f:
                scores = model.predict(feats[p.id])
                ranked = np.argsort(-scores, kind="stable")
                picks[p.id] = [cands[p.id][k] for k in ranked]
        progress(f"fold {f + 1}/{folds} trained on {len(group)} profiles")

    # The fair rule baseline: the same candidates with the restricted-program check applied,
    # since the reranker gets that check as a feature and the eligibility filter lacks it.
    rules = {
        p.id: [
            i
            for i in cands[p.id]
            if restriction(postings[i].title, postings[i].description) is None
        ]
        for p in profiles
    }
    # Judge whatever either new list put in its top 10 that nobody has graded yet.
    tops = {p.id: set(picks[p.id][:10]) | set(rules[p.id][:10]) for p in profiles}
    todo = [
        (p, postings[i])
        for p in profiles
        for i in tops[p.id]
        if not (hasattr(judge, "is_cached") and judge.is_cached(p, postings[i]))
    ]
    calls_before = judge.calls
    preflight(judge, todo, progress)
    base_scores, rule_scores, ltr_scores = [], [], []
    for p in profiles:
        qrels: dict[str, int] = {}
        for k, i in enumerate(cands[p.id]):
            if i in tops[p.id] or labels[p.id][k] is not None:
                g = judge.grade(p, postings[i]).grade
                if g is not None:
                    qrels[postings[i].id] = g
        base_scores.append(ndcg_at_k([postings[i].id for i in cands[p.id]], qrels, 10))
        rule_scores.append(ndcg_at_k([postings[i].id for i in rules[p.id]], qrels, 10))
        ltr_scores.append(ndcg_at_k([postings[i].id for i in picks[p.id]], qrels, 10))
    return LtrResult(
        profiles=len(profiles),
        folds=folds,
        baseline=base_scores,
        rules=rule_scores,
        ltr=ltr_scores,
        new_judgments=judge.calls - calls_before,
        importance=dict(zip(FEATURES, gain.tolist(), strict=True)),
    )

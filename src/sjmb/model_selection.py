"""Model selection for the reranker: complexity sweep, learning curve, feature ablation, a
linear baseline with rescaled coefficients, and ROC curves. Uses only cached judge grades.

Every model is scored with 5-fold cross-validation by profile (a student's pairs never sit in
both the training and the test fold), with folds stratified by career track. Each profile is
scored on its judged candidates: the model reorders them and NDCG@10 is computed against the
judge's grades. Unjudged candidates are left out, so no new judge calls are needed; this is a
reranking-of-the-judged-pool evaluation, not the pool-expansion one in `ltr.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from sjmb.data.extract import restriction
from sjmb.eval.metrics import ndcg_at_k
from sjmb.filters.eligibility import eligibility_mask
from sjmb.judge.rubric import RELEVANT_MIN
from sjmb.ltr import CANDIDATES, FEATURES, features
from sjmb.retrievers import BM25, Dense, Hybrid, make_encoder, top_k
from sjmb.types import Posting, Profile

LEAVES = (2, 4, 8, 15, 31, 63)
ROUNDS = (25, 50, 100, 200, 400)
DEFAULT = {"leaves": 15, "rounds": 200}


@dataclass
class Group:
    profile: Profile
    X: np.ndarray  # judged candidates only, in hybrid order
    y: np.ndarray  # judge grades 0-4
    restricted: np.ndarray  # restricted-program flag per candidate


def build_groups(
    postings: list[Posting], profiles: list[Profile], judge, as_of: date
) -> list[Group]:
    bm25 = BM25().fit(postings)
    dense = Dense(make_encoder("lsa")).fit(postings)
    hybrid = Hybrid([bm25, dense]).fit(postings)
    groups = []
    for prof in profiles:
        q = prof.query()
        sb, sd, sh = bm25.scores(q), dense.scores(q), hybrid.scores(q)
        idx = list(top_k(sh, CANDIDATES, eligibility_mask(prof, postings, as_of)))
        rows, ys, rs = [], [], []
        for r, i in enumerate(idx, 1):
            if not judge.is_cached(prof, postings[i]):
                continue
            g = judge.grade(prof, postings[i]).grade
            if g is None:
                continue
            rows.append(features(prof, postings[i], sb[i], sd[i], sh[i], r))
            ys.append(g)
            rs.append(restriction(postings[i].title, postings[i].description) is not None)
        if len(ys) >= 2:
            groups.append(Group(prof, np.array(rows), np.array(ys), np.array(rs)))
    return groups


def stratified_folds(groups: list[Group], k: int = 5, seed: int = 0) -> list[int]:
    """Fold per group, dealt round-robin within each track so every fold has the same mix."""
    rng = np.random.default_rng(seed)
    fold = [0] * len(groups)
    by_track: dict[str, list[int]] = {}
    for n, g in enumerate(groups):
        by_track.setdefault(g.profile.track, []).append(n)
    offset = 0
    for track in sorted(by_track):
        members = by_track[track]
        rng.shuffle(members)
        for j, n in enumerate(members):
            fold[n] = (j + offset) % k
        offset += len(members)
    return fold


def ndcg(groups: list[Group], scores: list[np.ndarray]) -> float:
    vals = []
    for g, s in zip(groups, scores, strict=True):
        order = np.argsort(-s, kind="stable")
        ids = [str(i) for i in order]
        vals.append(ndcg_at_k(ids, {str(i): int(v) for i, v in enumerate(g.y)}, 10))
    return float(np.mean(vals))


def fit_lgbm(train: list[Group], cols: list[int], leaves: int, rounds: int, seed: int = 0):
    import lightgbm as lgb

    data = lgb.Dataset(
        np.vstack([g.X[:, cols] for g in train]),
        label=np.concatenate([g.y for g in train]),
        group=[len(g.y) for g in train],
    )
    params = {
        "objective": "lambdarank",
        "learning_rate": 0.05,
        "num_leaves": leaves,
        "min_data_in_leaf": 10,
        "seed": seed,
        "deterministic": True,
        "num_threads": 1,
        "verbose": -1,
    }
    model = lgb.train(params, data, num_boost_round=rounds)
    return lambda X: model.predict(X[:, cols])


def fit_linear(train: list[Group], cols: list[int], ridge: float = 1.0):
    """Least-squares fit of the grade on features rescaled to [0, 1] with train-fold min/max."""
    X = np.vstack([g.X[:, cols] for g in train])
    y = np.concatenate([g.y for g in train]).astype(float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)

    def scale(A):
        return np.clip((A[:, cols] - lo) / span, 0, 1)

    Z = np.hstack([scale(np.vstack([g.X for g in train])), np.ones((len(y), 1))])
    reg = ridge * np.eye(Z.shape[1])
    reg[-1, -1] = 0
    w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    return (lambda A: np.hstack([scale(A), np.ones((len(A), 1))]) @ w), w[:-1]


def cv(groups, fold, fit, k=5):
    """Mean train and test NDCG@10 across folds, plus out-of-fold scores per group."""
    tr, te, oof = [], [], [None] * len(groups)
    for f in range(k):
        train = [g for g, x in zip(groups, fold, strict=True) if x != f]
        test_idx = [n for n, x in enumerate(fold) if x == f]
        model = fit(train)
        tr.append(ndcg(train, [model(g.X) for g in train]))
        test = [groups[n] for n in test_idx]
        s = [model(g.X) for g in test]
        te.append(ndcg(test, s))
        for n, sc in zip(test_idx, s, strict=True):
            oof[n] = sc
    return float(np.mean(tr)), float(np.mean(te)), float(np.std(te)), oof


def roc(scores: np.ndarray, labels: np.ndarray, points: int = 60) -> tuple[list, float]:
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.concatenate([[0], tp / max(1, y.sum())])
    fpr = np.concatenate([[0], fp / max(1, (1 - y).sum())])
    auc = float(np.trapezoid(tpr, fpr))
    step = max(1, len(tpr) // points)
    curve = [
        [round(float(a), 4), round(float(b), 4)]
        for a, b in zip(fpr[::step], tpr[::step], strict=True)
    ]
    curve.append([1.0, 1.0])
    return curve, auc


def run(groups: list[Group], progress=lambda _: None) -> dict:
    fold = stratified_folds(groups)
    all_cols = list(range(len(FEATURES)))
    out: dict = {
        "profiles": len(groups),
        "pairs": int(sum(len(g.y) for g in groups)),
        "relevant_share": float(np.mean(np.concatenate([g.y for g in groups]) >= RELEVANT_MIN)),
        "features": list(FEATURES),
    }

    # Baselines on the same judged candidates.
    rank_col = FEATURES.index("hybrid_rrf")
    out["baselines"] = {
        "hybrid order": ndcg(groups, [g.X[:, rank_col] for g in groups]),
        "hybrid order, restricted programs last": ndcg(
            groups, [g.X[:, rank_col] - 10 * g.restricted for g in groups]
        ),
    }

    # 1. Complexity sweep: leaves x rounds, train vs test.
    sweep = []
    for leaves in LEAVES:
        for rounds in ROUNDS:
            tr, te, sd, _ = cv(
                groups, fold, lambda t, L=leaves, R=rounds: fit_lgbm(t, all_cols, L, R)
            )
            sweep.append(
                {"leaves": leaves, "rounds": rounds, "train": tr, "test": te, "test_sd": sd}
            )
            progress(f"sweep leaves={leaves} rounds={rounds}: train {tr:.3f} test {te:.3f}")
    out["sweep"] = sweep
    best = max(sweep, key=lambda r: r["test"])
    out["best"] = {k: best[k] for k in ("leaves", "rounds", "train", "test", "test_sd")}
    out["default"] = next(
        {k: r[k] for k in ("leaves", "rounds", "train", "test", "test_sd")}
        for r in sweep
        if r["leaves"] == DEFAULT["leaves"] and r["rounds"] == DEFAULT["rounds"]
    )
    L, R = best["leaves"], best["rounds"]

    # 2. Learning curve: share of training profiles used, best complexity.
    rng = np.random.default_rng(1)
    curve = []
    for share in (0.1, 0.25, 0.5, 0.75, 1.0):
        trs, tes = [], []
        for f in range(5):
            train = [g for g, x in zip(groups, fold, strict=True) if x != f]
            keep = rng.permutation(len(train))[: max(5, int(share * len(train)))]
            sub = [train[i] for i in keep]
            model = fit_lgbm(sub, all_cols, L, R)
            trs.append(ndcg(sub, [model(g.X) for g in sub]))
            test = [g for g, x in zip(groups, fold, strict=True) if x == f]
            tes.append(ndcg(test, [model(g.X) for g in test]))
        curve.append({"share": share, "train": float(np.mean(trs)), "test": float(np.mean(tes))})
        progress(f"learning curve {share:.0%}: test {np.mean(tes):.3f}")
    out["learning_curve"] = curve

    # 3. Ablation: drop one feature at a time, best complexity.
    _, full_te, _, lgbm_oof = cv(groups, fold, lambda t: fit_lgbm(t, all_cols, L, R))
    ablation = []
    for j, name in enumerate(FEATURES):
        cols = [c for c in all_cols if c != j]
        _, te, _, _ = cv(groups, fold, lambda t, c=cols: fit_lgbm(t, c, L, R))
        ablation.append({"feature": name, "test_without": te, "drop": full_te - te})
    out["ablation"] = sorted(ablation, key=lambda a: -a["drop"])
    progress("ablation done")

    # 4. Linear baseline on rescaled features.
    tr, te, sd, lin_oof = cv(groups, fold, lambda t: fit_linear(t, all_cols)[0])
    _, w = fit_linear(groups, all_cols)
    out["linear"] = {
        "train": tr,
        "test": te,
        "test_sd": sd,
        "coefficients": sorted(
            [{"feature": n, "coef": float(c)} for n, c in zip(FEATURES, w, strict=True)],
            key=lambda c: -abs(c["coef"]),
        ),
    }

    # 5. ROC for "relevant" (grade >= 3) on out-of-fold scores.
    labels = np.concatenate([(g.y >= RELEVANT_MIN).astype(int) for g in groups])
    rocs = {}
    for name, scores in (
        ("LambdaMART", np.concatenate(lgbm_oof)),
        ("Linear (rescaled)", np.concatenate(lin_oof)),
        ("Hybrid retrieval score", np.concatenate([g.X[:, rank_col] for g in groups])),
    ):
        pts, auc = roc(scores, labels)
        rocs[name] = {"points": pts, "auc": auc}
    out["roc"] = rocs
    return out


def save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))

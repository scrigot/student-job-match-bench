"""Judge-human agreement: Cohen's kappa, unweighted and weighted.

Use quadratic weights for the 0-4 scale: a 3-vs-4 disagreement should cost far less than 0-vs-4.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np


def cohen_kappa(
    a: list[int],
    b: list[int],
    n_labels: int = 5,
    weights: Literal[None, "linear", "quadratic"] = None,
) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    bad = [x for x in (*a, *b) if not 0 <= x < n_labels]
    if bad:
        raise ValueError(f"labels must be in [0, {n_labels - 1}], got {sorted(set(bad))}")
    observed = np.zeros((n_labels, n_labels))
    for x, y in zip(a, b, strict=True):
        observed[x, y] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((n_labels, n_labels))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / (n_labels - 1)
    elif weights == "quadratic":
        w = ((i - j) / (n_labels - 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    denom = (w * expected).sum()
    if denom == 0:
        return 1.0  # both raters used one identical label throughout
    return float(1 - (w * observed).sum() / denom)


def read_golden(path: Path) -> list[dict]:
    """Golden set CSV: profile_id, posting_id, grade (0-4), optional note."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["grade"] = int(r["grade"])
    return rows

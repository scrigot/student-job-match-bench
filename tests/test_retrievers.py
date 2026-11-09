import numpy as np
from conftest import make_posting

from sjmb.retrievers import BM25, Dense, Hybrid, rrf, top_k
from sjmb.retrievers.dense import LSAEncoder

POSTS = [
    make_posting(id="a", title="Quantitative Trader Intern", description="probability options"),
    make_posting(id="b", title="Software Engineer Intern", description="Python distributed"),
    make_posting(id="c", title="Investment Banking Summer Analyst", description="M&A modeling"),
    make_posting(id="d", title="Data Scientist Intern", description="Python machine learning"),
    make_posting(id="e", title="Software Engineer", description="Java backend services"),
]


def test_bm25_ranks_the_obvious_match_first():
    bm = BM25().fit(POSTS)
    assert int(np.argmax(bm.scores("investment banking M&A analyst"))) == 2
    assert int(np.argmax(bm.scores("quantitative trader options"))) == 0
    assert bm.scores("zzzz unseen").sum() == 0


def test_lsa_dense_is_normalized_and_ranks_sensibly():
    dense = Dense(LSAEncoder(dim=4)).fit(POSTS)
    norms = np.linalg.norm(dense.matrix, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)
    assert int(np.argmax(dense.scores("software engineer python"))) in (1, 4)


def test_rrf_rewards_agreement_and_ignores_scale():
    a = np.array([100.0, 50.0, 1.0])
    b = np.array([0.2, 0.9, 0.1])
    fused = rrf([a, b], 3, k=60)
    assert fused.argmax() in (0, 1) and fused[2] == fused.min()
    assert np.allclose(rrf([a * 1000, b], 3), fused)  # scale-invariant
    # Unmatched documents (score 0) get no credit, so index order cannot leak in.
    z = rrf([np.array([0.0, 0.0, 2.0])], 3)
    assert z[0] == z[1] == 0 and z[2] > 0


def test_hybrid_and_top_k_with_mask():
    bm = BM25().fit(POSTS)
    dense = Dense(LSAEncoder(dim=4)).fit(POSTS)
    hyb = Hybrid([bm, dense]).fit(POSTS)
    s = hyb.scores("software engineer intern python")
    assert top_k(s, 1)[0] == 1
    mask = np.array([True, False, True, True, True])
    ranked = top_k(s, 5, mask)
    assert 1 not in ranked and len(ranked) == 4
    assert top_k(s, 3, np.zeros(5, dtype=bool)) == []

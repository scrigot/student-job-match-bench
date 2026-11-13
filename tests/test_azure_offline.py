"""Azure code paths with a fake OpenAI client: no network, no keys, no cost."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import AS_OF, make_posting, make_profile

from sjmb.azure import estimate_cost, v1_base_url
from sjmb.eval.runner import preflight
from sjmb.judge.judges import AzureJudge, BudgetExceeded
from sjmb.retrievers.dense import AzureEmbeddingEncoder


class FakeLLM:
    def __init__(self, reply='{"grade": 3, "reason": "fits"}', dim=8):
        self.chat_calls = 0
        self.embed_calls = 0
        self.reply, self.dim = reply, dim
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _chat(self, **kw):
        self.chat_calls += 1
        msg = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def _embed(self, model, input):
        self.embed_calls += 1
        data = [
            SimpleNamespace(index=i, embedding=list(np.random.default_rng(len(t)).random(self.dim)))
            for i, t in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))  # out of order on purpose


@pytest.fixture
def azure_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://res.services.ai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_AI_KEY", "test-not-a-key")
    monkeypatch.setenv("AZURE_JUDGE_DEPLOYMENT", "gpt-4.1-mini")
    monkeypatch.setenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")
    monkeypatch.setenv("SJMB_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_base_url_accepts_both_forms(azure_env, monkeypatch):
    assert v1_base_url() == "https://res.services.ai.azure.com/openai/v1"
    monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://res.services.ai.azure.com")
    assert v1_base_url() == "https://res.services.ai.azure.com/openai/v1"


def test_judge_caches_on_disk_and_never_stores_the_key(azure_env):
    llm = FakeLLM()
    j = AzureJudge(AS_OF, max_calls=10, llm=llm)
    prof, post = make_profile(), make_posting()
    assert j.grade(prof, post).grade == 3
    assert j.grade(prof, post).grade == 3
    assert llm.chat_calls == 1 and j.calls == 1
    # A fresh judge reads the cache from disk and makes no call.
    llm2 = FakeLLM()
    j2 = AzureJudge(AS_OF, max_calls=10, llm=llm2)
    assert j2.is_cached(prof, post) and j2.grade(prof, post).cached and llm2.chat_calls == 0
    assert "test-not-a-key" not in (azure_env / "judgments.jsonl").read_text()


def test_judge_budget_cap(azure_env):
    j = AzureJudge(AS_OF, max_calls=1, llm=FakeLLM())
    j.grade(make_profile(), make_posting(id="a"))
    with pytest.raises(BudgetExceeded):
        j.grade(make_profile(), make_posting(id="b"))


def test_preflight_stops_before_any_paid_call(azure_env, capsys):
    llm = FakeLLM()
    j = AzureJudge(AS_OF, max_calls=600, llm=llm)
    pairs = [(make_profile(), make_posting(id=str(i))) for i in range(700)]
    msgs = []
    with pytest.raises(SystemExit, match="700 uncached judge calls"):
        preflight(j, pairs, msgs.append)
    assert llm.chat_calls == 0
    assert msgs and "estimated $" in msgs[0]
    preflight(j, pairs[:400], msgs.append)  # under both thresholds: silent, no error
    assert len(msgs) == 1


def test_cost_estimate_uses_price_table():
    assert estimate_cost("gpt-4.1-mini", 1_000_000, 1_000_000) == pytest.approx(2.0)
    assert estimate_cost("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)


def test_embedding_cache(azure_env):
    llm = FakeLLM()
    enc = AzureEmbeddingEncoder(batch=2, llm=llm)
    texts = ["alpha", "beta", "gamma"]
    assert enc.estimate(texts)[0] == 2
    a = enc.encode(texts)
    assert a.shape == (3, 8) and llm.embed_calls == 2
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    b = AzureEmbeddingEncoder(batch=2, llm=llm).encode(["gamma", "alpha"])
    assert llm.embed_calls == 2  # all cached
    assert np.allclose(b, a[[2, 0]])


def test_unparsable_reply_is_retried_then_cached_as_none(azure_env):
    llm = FakeLLM(reply="")
    j = AzureJudge(AS_OF, max_calls=10, llm=llm)
    prof, post = make_profile(), make_posting()
    assert j.grade(prof, post).grade is None and llm.chat_calls == 2
    j2 = AzureJudge(AS_OF, max_calls=10, llm=llm)
    assert j2.grade(prof, post).grade is None and llm.chat_calls == 2  # not paid again


def test_cache_key_includes_as_of(azure_env):
    from datetime import date

    j = AzureJudge(AS_OF, max_calls=10, llm=FakeLLM())
    j.grade(make_profile(), make_posting())
    later = AzureJudge(date(2027, 1, 1), max_calls=10, llm=FakeLLM())
    assert not later.is_cached(make_profile(), make_posting())


def test_hard_check_runs_before_the_cache(azure_env):
    llm = FakeLLM()  # always answers grade 3
    j = AzureJudge(AS_OF, max_calls=10, llm=llm)
    prof, post = make_profile(grad_year=2029), make_posting(grad_years=[2028])
    assert j.grade(prof, post).grade == 0 and llm.chat_calls == 0  # code decided, no call


def test_code_decided_pairs_count_as_free(azure_env):
    j = AzureJudge(AS_OF, max_calls=0, llm=FakeLLM())
    prof = make_profile(grad_year=2028)
    closed = make_posting(grad_years=[2030])  # the code rules it out; no model call
    assert j.is_cached(prof, closed)
    preflight(j, [(prof, closed)], lambda _: None)  # no budget error at max_calls=0
    assert not j.is_cached(prof, make_posting())

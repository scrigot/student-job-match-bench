"""Azure AI Foundry access: client, on-disk caches and cost estimates.

Settings come from the environment (loaded from the git-ignored `.env`). Keys are never logged
or written to any cache.
"""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

# USD per 1M tokens, list prices for Global Standard deployments. Estimates only: check the
# portal's pricing page and update these if the deployments change.
PRICES = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}
DEFAULT_PRICE = {"input": 2.00, "output": 8.00}  # deliberately high for unknown deployments
COST_WARN_CALLS = 500  # print a cost estimate before any run with more uncached calls than this


def v1_base_url() -> str:
    """The OpenAI v1 route of the Foundry resource.

    Accepts AZURE_AI_ENDPOINT with or without the `/openai/v1` suffix, e.g.
    `https://<resource>.services.ai.azure.com` or `https://<resource>.services.ai.azure.com/openai/v1`.
    """
    endpoint = os.environ["AZURE_AI_ENDPOINT"].strip().rstrip("/")
    return endpoint.removesuffix("/openai/v1") + "/openai/v1"


def client() -> OpenAI:
    key = os.environ["AZURE_AI_KEY"]
    # Foundry accepts the key as `api-key`; the SDK also sends it as a bearer token.
    return OpenAI(
        base_url=v1_base_url(),
        api_key=key,
        default_headers={"api-key": key},
        max_retries=5,
        timeout=90,
    )


def cache_root() -> Path:
    root = Path(os.environ.get("SJMB_CACHE_DIR") or Path.home() / ".cache/student-job-match-bench")
    root.mkdir(parents=True, exist_ok=True)
    return root


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 characters per token for English


def estimate_cost(deployment: str, input_tokens: int, output_tokens: int = 0) -> float:
    price = PRICES.get(deployment, DEFAULT_PRICE)
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6

"""LLM cost estimation via OpenRouter's models endpoint.

OpenRouter exposes per-model pricing at GET /api/v1/models in the form
{"data": [{"id": "...", "pricing": {"prompt": "0.00000015", "completion": "..."}}]}
where the values are USD per token (yes, per token). We fetch this once,
cache it, and use it to compute total spend from per-model token counts.

For non-OpenRouter endpoints we fall back to a tiny hand-maintained table
covering the providers' native model IDs.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# (USD per 1M input tokens, USD per 1M output tokens). Used only when the
# endpoint isn't OpenRouter (which has its own dynamic pricing API).
_FALLBACK_PRICES_PER_M: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o":            (2.50, 10.00),
    "gpt-4o-mini":       (0.15,  0.60),
    "gpt-4.1":           (2.00,  8.00),
    "gpt-4.1-mini":      (0.40,  1.60),
    "o1":                (15.00, 60.00),
    "o3-mini":           (1.10,  4.40),
    # Anthropic native
    "claude-sonnet-4":   (3.00, 15.00),
    "claude-opus-4":     (15.00, 75.00),
    "claude-haiku-4":    (0.80,  4.00),
}

_openrouter_cache: Optional[Dict[str, Tuple[float, float]]] = None


def _fetch_openrouter_prices() -> Dict[str, Tuple[float, float]]:
    """Fetch per-model (input_per_M, output_per_M) prices from OpenRouter."""
    global _openrouter_cache
    if _openrouter_cache is not None:
        return _openrouter_cache

    try:
        import urllib.request
        import json
        with urllib.request.urlopen(
            "https://openrouter.ai/api/v1/models", timeout=10
        ) as r:
            data = json.loads(r.read())
    except Exception as e:
        logger.warning("Failed to fetch OpenRouter prices: %s", e)
        _openrouter_cache = {}
        return _openrouter_cache

    prices: Dict[str, Tuple[float, float]] = {}
    for model in data.get("data", []):
        pricing = model.get("pricing") or {}
        try:
            prompt = float(pricing.get("prompt", 0))      # USD per token
            completion = float(pricing.get("completion", 0))
        except (TypeError, ValueError):
            continue
        # Convert to USD per 1M tokens
        prices[model["id"]] = (prompt * 1_000_000, completion * 1_000_000)

    _openrouter_cache = prices
    logger.info("Loaded OpenRouter pricing for %d models", len(prices))
    return prices


def get_price(model_name: str, api_base: str) -> Optional[Tuple[float, float]]:
    """Return (input_per_M_usd, output_per_M_usd) for a model, or None if unknown."""
    api_base_lower = (api_base or "").lower()
    if "openrouter.ai" in api_base_lower:
        prices = _fetch_openrouter_prices()
        if model_name in prices:
            return prices[model_name]
        # Try the canonical "provider/model" id even if user passed just the suffix
        for k, v in prices.items():
            if k.endswith("/" + model_name):
                return v
        return None

    # Fallback table — strip provider prefix if present (e.g. "openai/gpt-4o-mini")
    short = model_name.split("/", 1)[-1]
    return _FALLBACK_PRICES_PER_M.get(short)


def estimate_cost(
    tokens_by_model: Dict[str, Tuple[int, int]],
    api_base: str,
) -> Tuple[float, list]:
    """Compute total USD cost.

    Args:
        tokens_by_model: {model_name: (input_tokens, output_tokens)}.
        api_base: API endpoint (used to pick OpenRouter vs fallback prices).

    Returns:
        (total_usd, unknown_models) where unknown_models is a list of model
        names whose price could not be resolved (their tokens contribute 0
        to the total).
    """
    total = 0.0
    unknown: list = []
    for model, (in_tok, out_tok) in tokens_by_model.items():
        price = get_price(model, api_base)
        if price is None:
            unknown.append(model)
            continue
        in_per_m, out_per_m = price
        total += (in_tok * in_per_m + out_tok * out_per_m) / 1_000_000
    return total, unknown

"""Local, explicitly configured cost estimates from model-returned token usage."""

from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation
from functools import wraps
import json
from pathlib import Path
from uuid import uuid4

_scope = ContextVar("inference_scope", default=None)


def load_pricing(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "pricing.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def estimate_cost(input_tokens, output_tokens, cached_tokens, pricing):
    """Return Decimal USD costs, or None if rates/usage are incomplete or invalid.

    Cached tokens are a subset of input tokens. Unknown cache counts get no
    discount (all input charged at the configured regular rate).
    """
    try:
        if not isinstance(pricing, dict) or pricing.get("currency") != "USD":
            return None
        if any(not pricing.get(key) for key in ("model", "deployment_type", "source", "version")):
            return None
        if any(type(n) is not int or n < 0 for n in (input_tokens, output_tokens)):
            return None
        cached = 0 if cached_tokens is None else cached_tokens
        if type(cached) is not int or not 0 <= cached <= input_tokens:
            return None
        regular = Decimal(str(pricing["input_per_million"]))
        output = Decimal(str(pricing["output_per_million"]))
        cache_rate = Decimal(str(pricing.get("cached_input_per_million")
                                 if pricing.get("cached_input_per_million") is not None else regular))
        if any(not p.is_finite() or p < 0 for p in (regular, output, cache_rate)):
            return None
        input_cost = ((input_tokens - cached) * regular + cached * cache_rate) / 1_000_000
        output_cost = output_tokens * output / 1_000_000
        return input_cost, output_cost, input_cost + output_cost
    except (KeyError, ValueError, TypeError, InvalidOperation):
        return None


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


class InferenceMetrics:
    def __init__(self, pricing=None):
        self.pricing = load_pricing() if pricing is None else pricing
        if not isinstance(self.pricing, dict):
            self.pricing = {}
        self.session_calls = []
        self.query_calls = []
        self._seen = set()
        self.query_id = str(uuid4())

    def reset_query(self):
        self.query_calls = []
        self.query_id = str(uuid4())

    def record(self, raw, capability, deployment, *, call_id=None, kind="llm"):
        response_id = _get(raw, "id")
        identity = (deployment, response_id or call_id or str(uuid4()))
        if identity in self._seen:
            return False
        self._seen.add(identity)
        usage = _get(raw, "usage")
        input_tokens = _get(usage, "input_tokens", _get(usage, "prompt_tokens"))
        output_tokens = _get(usage, "output_tokens", 0 if kind == "embedding" and usage else None)
        cached = _get(_get(usage, "input_tokens_details"), "cached_tokens")
        deployments = self.pricing.get("deployments", {})
        rates = deployments.get(deployment) if isinstance(deployments, dict) else None
        costs = estimate_cost(input_tokens, output_tokens, cached, rates)
        record = {
            "call_id": identity[1], "query_id": self.query_id,
            "capability": capability, "model_id": deployment,
            "returned_model_id": _get(raw, "model"), "kind": kind,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cached_input_tokens": cached,
            "estimated_input_cost_usd": str(costs[0]) if costs else None,
            "estimated_output_cost_usd": str(costs[1]) if costs else None,
            "estimated_inference_cost_usd": str(costs[2]) if costs else None,
            "pricing": dict(rates) if isinstance(rates, dict) else None,
            "pricing_config_version": self.pricing.get("version"),
            "cost_status": "estimated" if costs else "unavailable",
        }
        self.query_calls.append(record)
        self.session_calls.append(record)
        return True

    @staticmethod
    def totals(calls):
        costs = [c["estimated_inference_cost_usd"] for c in calls]
        return {
            "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
            "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
            "llm_call_count": sum(c["kind"] == "llm" for c in calls),
            "llm_input_tokens": sum(c["input_tokens"] or 0 for c in calls if c["kind"] == "llm"),
            "llm_output_tokens": sum(c["output_tokens"] or 0 for c in calls if c["kind"] == "llm"),
            "embedding_call_count": sum(c["kind"] == "embedding" for c in calls),
            "model_call_count": len(calls),
            "usage_complete": all(c["input_tokens"] is not None and c["output_tokens"] is not None for c in calls),
            "estimated_inference_cost_usd": (
                str(sum((Decimal(c) for c in costs), Decimal(0))) if all(c is not None for c in costs) else None
            ),
        }

    def snapshot(self):
        return {"query_id": self.query_id, "query": self.totals(self.query_calls),
                "session": self.totals(self.session_calls), "calls": list(self.query_calls)}


@contextmanager
def usage_scope(state, capability):
    token = _scope.set((state, capability))
    try:
        yield
    finally:
        _scope.reset(token)


def instrument_capability(function):
    """Bind each sequential executor's calls to its logical capability."""
    @wraps(function)
    async def wrapped(self, message, ctx):
        executor = getattr(self, "id", "unknown")
        capability = executor if executor in ("parser", "orchestrator") else getattr(message, "agent_name", executor)
        with usage_scope(message.conversation_state, capability):
            return await function(self, message, ctx)
    return wrapped


def record_response(raw, deployment, kind="llm"):
    current = _scope.get()
    if current is not None and current[0] is not None:
        state, capability = current
        if state.inference_metrics.record(raw, capability, deployment, kind=kind):
            totals = state.inference_metrics.totals(state.inference_metrics.query_calls)
            state.input_tokens = totals["input_tokens"]
            state.output_tokens = totals["output_tokens"]


def instrument_embeddings(embedding_function, deployment):
    """Capture the synchronous SDK response before Chroma discards usage."""
    create = embedding_function.client.embeddings.create
    @wraps(create)
    def metered_create(*args, **kwargs):
        raw = create(*args, **kwargs)
        record_response(raw, deployment, kind="embedding")
        return raw
    embedding_function.client.embeddings.create = metered_create

"""Offline arithmetic and SDK-boundary tests; not Azure billing validation."""
import csv
import json
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import httpx
from openai import AsyncOpenAI
from agents2.inference_metrics import InferenceMetrics, estimate_cost, usage_scope, instrument_embeddings
from agents2.metered_client import MeteredResponsesClient
from agents2.shared_types import ConversationState
from query_logger3 import LEGACY_COLUMNS, log_query, log_turn_metrics

# Deliberately synthetic rates, not Azure prices.
RATES = dict(model="test-model", deployment_type="test", currency="USD",
             source="synthetic test fixture", version="test-v1",
             input_per_million=2, output_per_million=8, cached_input_per_million=0.5)
CONFIG = {"version": "test-v1", "deployments": {"test-deployment": RATES}}


def response(identity="r1", inputs=1000, outputs=100, cached=None):
    usage = dict(input_tokens=inputs, output_tokens=outputs)
    if cached is not None:
        usage["input_tokens_details"] = {"cached_tokens": cached}
    return dict(id=identity, model="test-model", usage=usage)


class CostTests(unittest.TestCase):
    def test_known_counts(self):
        self.assertEqual(estimate_cost(1000, 100, None, RATES),
                         (Decimal(".002"), Decimal(".0008"), Decimal(".0028")))

    def test_cached_tokens_replace_regular_charge(self):
        self.assertEqual(estimate_cost(1000, 100, 400, RATES)[2], Decimal(".0022"))

    def test_unavailable_cache_count_is_preserved(self):
        metrics = InferenceMetrics(CONFIG)
        metrics.record(response(), "parser", "test-deployment")
        self.assertIsNone(metrics.query_calls[0]["cached_input_tokens"])
        metrics.record(response("r2", cached=0), "parser", "test-deployment")
        self.assertEqual(metrics.query_calls[1]["cached_input_tokens"], 0)

    def test_calls_and_queries_aggregate_without_session_reset(self):
        metrics = InferenceMetrics(CONFIG)
        metrics.record(response(), "parser", "test-deployment")
        metrics.record(response("r2"), "planning", "test-deployment")
        self.assertEqual(metrics.snapshot()["query"]["llm_call_count"], 2)
        self.assertEqual(Decimal(metrics.snapshot()["query"]["estimated_inference_cost_usd"]), Decimal(".0056"))
        old_id = metrics.query_id
        metrics.reset_query()
        self.assertNotEqual(metrics.query_id, old_id)
        self.assertEqual(metrics.snapshot()["query"]["input_tokens"], 0)
        self.assertEqual(metrics.snapshot()["query"]["llm_call_count"], 0)
        metrics.record(response("r3"), "orchestrator", "test-deployment")
        self.assertEqual(metrics.snapshot()["session"]["input_tokens"], 3000)
        self.assertEqual(metrics.snapshot()["session"]["output_tokens"], 300)
        self.assertEqual(metrics.snapshot()["session"]["llm_call_count"], 3)
        self.assertEqual(Decimal(metrics.snapshot()["session"]["estimated_inference_cost_usd"]), Decimal(".0084"))

    def test_duplicate_response_not_counted_twice_across_turns(self):
        metrics = InferenceMetrics(CONFIG)
        self.assertTrue(metrics.record(response(), "parser", "test-deployment"))
        self.assertFalse(metrics.record(response(), "parser", "test-deployment"))
        metrics.reset_query()
        self.assertFalse(metrics.record(response(), "parser", "test-deployment"))
        self.assertEqual(metrics.snapshot()["session"]["llm_call_count"], 1)

    def test_unknown_or_bad_pricing_never_assumes_free_inference(self):
        for config in ({}, {"deployments": []}, {"deployments": {"test-deployment": "bad"}}):
            metrics = InferenceMetrics(config)
            metrics.record(response(), "parser", "test-deployment")
            self.assertEqual(metrics.snapshot()["query"]["input_tokens"], 1000)
            self.assertIsNone(metrics.snapshot()["query"]["estimated_inference_cost_usd"])
        self.assertIsNone(estimate_cost(1000, 100, 1001, RATES))
        self.assertIsNone(estimate_cost(1000, 100, None, dict(RATES, input_per_million="NaN")))

    def test_missing_usage_and_mixed_pricing_are_unavailable(self):
        metrics = InferenceMetrics(CONFIG)
        metrics.record(response(), "parser", "test-deployment")
        metrics.record({"id": "missing"}, "planning", "test-deployment")
        self.assertFalse(metrics.snapshot()["query"]["usage_complete"])
        self.assertIsNone(metrics.snapshot()["session"]["estimated_inference_cost_usd"])

    def test_state_reset_preserves_conversation_and_session(self):
        state = ConversationState(transcript_data={"completed_courses": []})
        state.add_message("user", "hello")
        state.resolved_courses = {"X": {"title": "X"}}
        state.inference_metrics.record(response(), "parser", "test-deployment")
        state.routing_ctx = object()
        state.routing_iteration = 4
        state.routing_events = [{}]
        state.reset_usage()
        self.assertIsNone(state.routing_ctx)
        self.assertEqual(state.routing_events, [])
        self.assertEqual(state.routing_iteration, 0)
        self.assertEqual(state.inference_metrics.snapshot()["session"]["llm_call_count"], 1)
        self.assertTrue(state.transcript_data)
        self.assertTrue(state.resolved_courses)
        self.assertEqual(len(state.conversation_history), 1)

    def test_embedding_usage_captured_separately(self):
        raw = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10), model="test-model")
        function = SimpleNamespace(client=SimpleNamespace(embeddings=SimpleNamespace(create=Mock(return_value=raw))))
        instrument_embeddings(function, "test-deployment")
        state = ConversationState()
        state.inference_metrics = InferenceMetrics(CONFIG)
        with usage_scope(state, "data_fetch"):
            self.assertIs(function.client.embeddings.create(input=["x"]), raw)
        snapshot = state.inference_metrics.snapshot()
        self.assertEqual(snapshot["query"]["llm_call_count"], 0)
        self.assertEqual(snapshot["query"]["model_call_count"], 1)
        self.assertEqual(snapshot["query"]["input_tokens"], 10)

    def test_legacy_csv_preserved_and_sidecars_join_by_query_id(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "log.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(LEGACY_COLUMNS)
                writer.writerow(["historical"] + [""] * (len(LEGACY_COLUMNS) - 1))
            original = path.read_bytes()
            metrics = InferenceMetrics(CONFIG)
            metrics.record(response(), "parser", "test-deployment")
            log_query("session", "q", "a", ["parser"], filepath=path,
                      model_id="test-deployment", research=metrics.snapshot())
            log_turn_metrics("session", metrics.snapshot(), filepath=path)
            self.assertTrue(path.read_bytes().startswith(original))
            with path.with_name("log_metrics.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["query_id"], metrics.query_id)
            self.assertEqual(row["topology"], "star")
            self.assertEqual(row["llm_call_count"], "1")
            detail = json.loads(path.with_name("log_calls.jsonl").read_text())
            self.assertEqual(detail["query_id"], row["query_id"])
            self.assertEqual(detail["calls"][0]["capability"], "parser")


class SDKBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_agent_overrides_are_sent_metered_priced_and_logged(self):
        sent_models = []
        def handle(request):
            sent_models.append(json.loads(request.content)["model"])
            body = response(f"mixed-{len(sent_models)}", cached=400)
            body.update(object="response", created_at=0, status="completed", output=[],
                        parallel_tool_calls=True, tool_choice="auto", tools=[])
            body["usage"].update(total_tokens=1100, output_tokens_details={"reasoning_tokens": 0})
            return httpx.Response(200, json=body)
        sdk = AsyncOpenAI(api_key="offline", base_url="https://offline.invalid/openai/v1",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        client = MeteredResponsesClient(async_client=sdk, model_id="must-not-be-used")
        state = ConversationState()
        state.inference_metrics = InferenceMetrics({"deployments": {
            "specialist": RATES,
            "router": dict(RATES, input_per_million=4, output_per_million=16, cached_input_per_million=1),
        }})
        config = {"parser": "specialist", "orchestrator": "router", "response_writer": "specialist"}
        try:
            for capability, deployment in config.items():
                agent = client.as_agent(name=capability, default_options={"model_id": deployment})
                with usage_scope(state, capability):
                    await agent.run("offline test")
            self.assertEqual(sent_models, ["specialist", "router", "specialist"])
            snapshot = state.inference_metrics.snapshot()
            self.assertEqual([c["model_id"] for c in snapshot["calls"]], sent_models)
            self.assertEqual([Decimal(c["estimated_inference_cost_usd"]) for c in snapshot["calls"]],
                             [Decimal(".0022"), Decimal(".0044"), Decimal(".0022")])
            self.assertEqual(snapshot["query"]["llm_call_count"], 3)
            with TemporaryDirectory() as directory:
                path = Path(directory) / "mixed.csv"
                log_query("s", "q", "a", ["parser", "orchestrator"], research=snapshot, filepath=path)
                log_turn_metrics("s", snapshot, model_config=config, filepath=path)
                with path.open(newline="", encoding="utf-8") as stream:
                    self.assertEqual(next(csv.DictReader(stream))["model_id"], "router|specialist")
                detail = json.loads(path.with_name("mixed_calls.jsonl").read_text())
                self.assertEqual(detail["model_config"], config)
        finally:
            await sdk.close()

    async def test_real_sdk_and_framework_parse_once_without_network(self):
        count = 0
        def handle(request):
            nonlocal count
            count += 1
            body = response(f"resp-{count}", cached=400)
            body.update(object="response", created_at=0, status="completed", output=[],
                        parallel_tool_calls=True, tool_choice="auto", tools=[])
            body["usage"]["total_tokens"] = 1100
            body["usage"]["output_tokens_details"] = {"reasoning_tokens": 0}
            return httpx.Response(200, json=body)
        http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        sdk = AsyncOpenAI(api_key="offline", base_url="https://offline.invalid/openai/v1", http_client=http)
        client = MeteredResponsesClient(async_client=sdk, model_id="test-deployment")
        state = ConversationState()
        state.inference_metrics = InferenceMetrics(CONFIG)
        try:
            with usage_scope(state, "parser"):
                await client.get_response("test")
            with usage_scope(state, "orchestrator"):
                await client.get_response("test")
            self.assertEqual(count, 2)
            self.assertEqual(state.input_tokens, 2000)
            self.assertEqual(state.inference_metrics.snapshot()["query"]["llm_call_count"], 2)
            self.assertEqual(Decimal(state.inference_metrics.snapshot()["query"]["estimated_inference_cost_usd"]), Decimal(".0044"))
        finally:
            await sdk.close()

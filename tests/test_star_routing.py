"""Regression checks for the existing STAR policy, not new routing behavior."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from driver3 import build_workflow
from agents2.orchestrator_agent import OrchestratorExecutor, RoutingContext, OrchestratorRequest
from agents2.shared_types import ConversationState


class StarRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def route(self, agents, *, mode="route", results=None):
        executor = OrchestratorExecutor(Mock(), "test", "writer-test")
        executor.agent = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
            "mode": mode, "next_agents": agents, "response": "answer", "reasoning": "test"}))))
        executor.response_agent = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(content="fallback")))
        state = ConversationState()
        routing = RoutingContext("recommend AI", {"intent": "course_recommendation"}, False,
                                 accumulated_results=results or {})
        message = OrchestratorRequest(routing.user_query, routing.parsed_data, state)
        ctx = SimpleNamespace(send_message=AsyncMock(), yield_output=AsyncMock())
        await executor._routing_loop(routing, message, ctx, 0)
        return state.routing_events[0], ctx

    async def test_valid_route_unchanged(self):
        event, ctx = await self.route(["data_fetch"])
        self.assertTrue(event["accepted_unchanged"])
        self.assertEqual(event["executed_agents"], ["data_fetch"])
        self.assertEqual(ctx.send_message.await_args.args[0].agent_name, "data_fetch")

    async def test_invalid_agent_repaired(self):
        event, _ = await self.route(["invented"])
        self.assertEqual(event["proposed_agents"], ["invented"])
        self.assertEqual(event["executed_agents"], ["data_fetch"])
        self.assertIn("invalid_agent", event["interventions"])
        self.assertFalse(event["accepted_unchanged"])

    async def test_duplicate_completed_agent_rejected(self):
        event, _ = await self.route(["data_fetch"], results={"data_fetch": {"courses": [{"code": "X"}]}})
        self.assertIn("duplicate_agent", event["interventions"])
        self.assertEqual(event["executed_agents"], ["planning"])

    async def test_downstream_dependency_repaired(self):
        event, _ = await self.route(["planning"])
        self.assertEqual(event["executed_agents"], ["data_fetch"])
        self.assertIn("missing_dependency", event["interventions"])

    async def test_early_response_blocked(self):
        event, ctx = await self.route([], mode="respond")
        self.assertEqual(event["proposed_mode"], "respond")
        self.assertEqual(event["executed_agents"], ["data_fetch"])
        self.assertIn("early_response_blocked", event["interventions"])
        ctx.yield_output.assert_not_awaited()

    async def test_empty_planning_payload_not_logged_as_executed(self):
        event, ctx = await self.route(["planning"], results={"data_fetch": {"courses": []}})
        self.assertEqual(event["executed_agents"], [])
        self.assertIn("empty_candidates", event["interventions"])
        ctx.send_message.assert_not_awaited()
        ctx.yield_output.assert_awaited_once_with("fallback")

    async def test_hybrid_intent_restriction_explicit(self):
        event, _ = await self.route(["data_prereq"])
        self.assertIn("intent_or_capability_block", event["interventions"])

    def test_only_star_is_supported(self):
        with self.assertRaisesRegex(ValueError, "Unsupported topology"):
            build_workflow(None, "test", topology="mesh")

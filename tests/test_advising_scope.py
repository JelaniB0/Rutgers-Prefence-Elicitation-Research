import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from agents2.shared_types import ConversationState
from agents2.orchestrator_agent import OrchestratorExecutor, OrchestratorRequest, RoutingContext


class ScopeTests(unittest.IsolatedAsyncioTestCase):
    def test_campus_removed_from_parser_and_pending_fields(self):
        state = ConversationState()
        parsed = {"intent": "prerequisite_check", "entities": {"campus": "NB"},
                  "missing_critical_info": ["target_course", "campus"],
                  "memory_updates": {"campus": "NB", "preferences": {"campus": "NB"}},
                  "suggested_clarifications": ["Which campus?", "Which course?"]}
        state.resume_clarification(parsed)
        state.enrich_parsed_query(parsed)
        state.request_clarification("How can I take machine learning?", parsed,
                                    ["target_course", "campus"], "Which course and campus?")
        self.assertNotIn("campus", json.dumps(parsed).lower())
        self.assertNotIn("campus", json.dumps(state.pending_clarification).lower())
        reply = {"intent": "course_info", "entities": {"target_course": "Machine Learning Principles"}}
        state.resume_clarification(reply)
        self.assertEqual(reply["intent"], "prerequisite_check")
        self.assertIsNone(state.pending_clarification)
        self.assertNotIn("clarification_question", reply)

    async def check_router(self, entities, missing):
        state = ConversationState()
        parsed = {"intent": "prerequisite_check", "entities": entities}
        hub = OrchestratorExecutor(Mock(), "router", "writer")
        hub.agent = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
            "mode": "clarify", "response": "Which course and campus?", "next_agents": [],
            "missing_fields": missing}))))
        routing = RoutingContext("How can I take machine learning?", parsed, False)
        ctx = SimpleNamespace(send_message=AsyncMock(), yield_output=AsyncMock())
        await hub._routing_loop(routing, OrchestratorRequest(routing.user_query, parsed, state), ctx, 0)
        return state, ctx

    async def test_clarification_asks_course_only(self):
        state, ctx = await self.check_router({}, ["target_course", "campus"])
        self.assertEqual(state.pending_clarification["missing_fields"], ["target_course"])
        self.assertNotIn("campus", ctx.yield_output.await_args.args[0].lower())

    async def test_known_course_routes_without_campus(self):
        state, ctx = await self.check_router({"target_course": "Machine Learning Principles"}, ["campus"])
        ctx.yield_output.assert_not_awaited()
        self.assertEqual(ctx.send_message.await_args.args[0].agent_name, "data_prereq")
        self.assertIsNone(state.pending_clarification)

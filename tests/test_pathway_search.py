import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from agents2.pathway_search import search_pathways, pathways_for_transcript
from agents2.dag_builder import check_eligibility
from agents2.orchestrator_agent import AgentResult, RoutingContext
from agents2.shared_types import AgentResponse, ConversationState
from driver3 import DataExecutor


def node(and_=(), or_=(), permission=False):
    return {"title": "Example course", "and": list(and_),
            "or_groups": list(or_), "requires_permission": permission}


class PathwaySearchTests(unittest.TestCase):
    def test_and_branches_are_all_required_and_can_run_in_parallel(self):
        dag = {"T": node(["A", "B"]), "A": node(["C"]), "B": node(["C"]), "C": node()}
        result = search_pathways(dag, "T")
        plan = result["plans"][0]
        self.assertEqual(plan["additional_course_count"], 3)
        self.assertEqual([[c["code"] for c in stage] for stage in plan["stages"]], [["C"], ["A", "B"]])
        self.assertTrue(result["shortest_proven"])
        self.assertIsNone(result["eligible_from_completed_courses"])

    def test_fewest_courses_is_not_always_fewest_rounds(self):
        dag = {"T": node(or_=[["A", "B"]]), "A": node(["C"]), "C": node(["D"]),
               "D": node(), "B": node(["E", "F", "G"]), "E": node(), "F": node(), "G": node()}
        result = search_pathways(dag, "T")
        self.assertEqual(result["plans"][0]["additional_course_count"], 3)
        fastest = result["plans"][result["fastest_examined_plan_index"]]
        self.assertEqual(fastest["prerequisite_rounds"], 2)
        self.assertEqual(fastest["additional_course_count"], 4)

    def test_in_progress_is_conditional_and_completed_work_is_not_repeated(self):
        dag = {"T": node(["A"]), "A": node(["B"]), "B": node()}
        result = search_pathways(dag, "T", completed={"B"}, in_progress={"A"}, has_transcript=True)
        self.assertFalse(result["eligible_from_completed_courses"])
        self.assertTrue(result["eligible_after_in_progress"])
        self.assertEqual(result["plans"][0]["additional_course_count"], 0)
        self.assertEqual(result["in_progress_assumed_passed"][0]["code"], "A")

    def test_unknown_and_permission_nodes_are_not_free_prerequisites(self):
        for dag in ({"T": node(["UNKNOWN"])}, {"T": node(permission=True)}):
            result = search_pathways(dag, "T")
            self.assertEqual(result["status"], "unresolved")
            self.assertFalse(result["plans"])
        self.assertFalse(check_eligibility("UNKNOWN", {}, set(), set())["eligible"])
        self.assertFalse(check_eligibility("T", {"T": node(permission=True)}, set(), set())["eligible"])

    def test_cycles_terminate_and_valid_or_branches_still_work(self):
        dag = {"T": node(or_=[["A", "B"]]), "A": node(["A"]), "B": node()}
        result = search_pathways(dag, "T")
        self.assertIn("A", result["cycle_nodes"])
        self.assertEqual(result["plans"][0]["stages"][0][0]["code"], "B")

    def test_partial_catalog_produces_explicitly_conditional_plan(self):
        result = search_pathways({"T": node(["A", "MATH"]), "A": node()}, "T")
        self.assertFalse(result["plans"])
        self.assertEqual(result["status"], "unresolved")
        partial = result["conditional_plans"][0]
        self.assertEqual(partial["external_requirements_to_verify"][0]["code"], "MATH")
        self.assertEqual(partial["stages"][0][0]["code"], "A")
        self.assertFalse(result["shortest_proven"])

    def test_search_budget_is_explicit(self):
        result = search_pathways({"T": node(["A"]), "A": node()}, "T", max_states=1)
        self.assertTrue(result["search_limited"])
        self.assertFalse(result["shortest_proven"])

    def test_failed_courses_are_excluded_and_transfer_credit_is_used(self):
        dag = {"T": node(["A", "B"]), "A": node(), "B": node()}
        result = pathways_for_transcript(dag, "T", {
            "completed_courses": [{"code": "A", "grade": "F"}],
            "transfer_courses": [{"code": "B"}],
        })
        self.assertEqual(result["plans"][0]["additional_course_count"], 1)
        self.assertEqual(result["plans"][0]["stages"][0][0]["code"], "A")


class PathwayIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_prerequisite_spoke_returns_paths_and_router_preserves_them(self):
        with TemporaryDirectory() as directory:
            dagfile = Path(directory) / "dag.json"
            dagfile.write_text(json.dumps({"T": node(["A"]), "A": node()}), encoding="utf-8")
            state = ConversationState(transcript_data={"completed_courses": []})
            message = AgentResult("How can I take T?", {"entities": {"target_course": "T"}},
                                  "data_prereq", {}, state)
            executor = SimpleNamespace(data_agent=SimpleNamespace(lookup_course=AsyncMock(
                return_value=AgentResponse(success=True, data={"course": {"code": "T"}})
            )), _pathways=DataExecutor._pathways)
            ctx = SimpleNamespace(send_message=AsyncMock(), yield_output=AsyncMock())
            with patch("driver3.DAG_FILE", dagfile):
                await DataExecutor.handle(executor, message, ctx)
            outgoing = ctx.send_message.call_args.args[0]
            self.assertEqual(outgoing.data["pathways"][0]["plans"][0]["additional_course_count"], 1)
            routing = RoutingContext("How can I take T?", {}, True,
                                     accumulated_results={"data_prereq": outgoing.data})
            self.assertIn("pathways", routing._slim_results()["data_prereq"])

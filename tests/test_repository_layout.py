"""Offline checks for resource paths and the driver's actual logging path."""

import csv
from contextlib import chdir
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import driver3
from agent_framework import WorkflowOutputEvent
from agents2.azure_openai import AzureOpenAISettings, ModelConfig, create_agent_client
from agents2.paths import COURSES_FILE, DAG_FILE, SCHEMA_FILE, QUERY_LOG_FILE
from query_logger3 import CSV_LOG_FILE, log_query, log_turn_metrics


class RepositoryLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_constructs_outside_repo(self):
        settings = AzureOpenAISettings(
            "https://example.openai.azure.com/openai/v1", "test-key",
            "text-embedding-3-small",
            ModelConfig("parser-model", "router-model", "data-model", "constraint-model",
                        "planning-model", "transcript-model", "writer-model"),
        )
        client = create_agent_client(settings)
        try:
            with TemporaryDirectory() as directory, chdir(directory), patch.object(
                driver3.DataAgent, "_initialize_vector_db", return_value=Mock()
            ), patch.object(driver3.DataAgent, "_index_courses"):
                workflow, orchestrator = driver3.build_workflow(client, settings.models)
                self.assertIsNotNone(workflow)
                edges = {(edge.source_id, edge.target_id)
                         for group in workflow.edge_groups for edge in group.edges
                         if not edge.source_id.startswith("internal:")}
                self.assertEqual(edges, {
                    ("parser", "orchestrator"),
                    ("orchestrator", "data"), ("data", "orchestrator"),
                    ("orchestrator", "constraint"), ("constraint", "orchestrator"),
                    ("orchestrator", "planning"), ("planning", "orchestrator"),
                    ("orchestrator", "transcript"),
                })
                self.assertEqual(orchestrator.agent.default_options["model_id"], settings.models.orchestrator)
                self.assertEqual(orchestrator.response_agent.default_options["model_id"], settings.models.response_writer)
                for role, attribute in (("parser", "parser"), ("data", "data_agent"),
                                        ("constraint", "constraint_agent"), ("planning", "planning_agent"),
                                        ("transcript", "transcript_agent")):
                    agent = getattr(workflow.executors[role], attribute)
                    self.assertEqual(agent.default_options["model_id"], getattr(settings.models, role))
                self.assertTrue(driver3.ConstraintAgent(client, "gpt-5").dag)
                for resource in (COURSES_FILE, DAG_FILE, SCHEMA_FILE):
                    self.assertTrue(resource.is_file(), str(resource))
                self.assertEqual(Path(CSV_LOG_FILE), QUERY_LOG_FILE)
                self.assertTrue(Path(CSV_LOG_FILE).is_absolute())
        finally:
            await client.client.close()

    async def test_driver_appends_recommendation_to_real_csv_logger(self):
        response = 'Course recommendation, with a comma\nand "quoted" details.'

        async def run_stream(message):
            state = message.conversation_state
            state.last_intent = "course_recommendation"
            state.routing_ctx = SimpleNamespace(agents_call_order=["data_fetch", "planning"])
            state.inference_metrics.record({"id": "test-response", "usage": {
                "input_tokens": 123, "output_tokens": 45}}, "parser", "gpt-5")
            yield WorkflowOutputEvent(response, executor_id="orchestrator")

        with TemporaryDirectory() as directory:
            logfile = Path(directory) / "results" / "query_log3.csv"
            log_query("earlier", "Existing query", "Existing answer", ["parser"], filepath=logfile)
            original = logfile.read_bytes()
            with chdir(directory), patch.object(driver3, "get_azure_openai_settings",
                return_value=SimpleNamespace(models=ModelConfig(*(["gpt-5"] * 7)))
            ), patch.object(driver3, "create_agent_client"), patch.object(
                driver3, "build_dag", new_callable=AsyncMock
            ), patch.object(driver3, "build_workflow", return_value=(
                SimpleNamespace(run_stream=run_stream), None
            )), patch.object(driver3, "log_turn_metrics", side_effect=partial(log_turn_metrics, filepath=logfile)), patch.object(driver3, "log_query", side_effect=partial(log_query, filepath=logfile)), patch.object(
                driver3, "_collect_feedback_and_hallucination",
                return_value=("yes", "", "no", "", "")
            ), patch("builtins.input", side_effect=["Recommend AI courses", "quit"]), patch("builtins.print"):
                await driver3.main()

            self.assertTrue(logfile.read_bytes().startswith(original))
            with logfile.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["response"], response)
            self.assertEqual(rows[1]["plan_steps"], "data_fetch -> planning")
            self.assertEqual(rows[1]["input_tokens"], "123")
            self.assertEqual(rows[1]["satisfied"], "yes")


if __name__ == "__main__":
    unittest.main()

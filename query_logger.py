"""
creates csv log of queries and their execution times, agents invoked, sources used. 
"""

import csv
import os
from datetime import datetime
from typing import Any

CSV_LOG_FILE = "query_log.csv"
 
CSV_COLUMNS = [
    "session_id",
    "timestamp",
    "query",
    "response",
    "agents_invoked",       # pipe-separated list of agent names, e.g. "ParserAgent|DataAgent|PlanningAgent"
    "sources_and_tools",    # structured detail of what each agent used, e.g. "DataAgent:rutgers_courses.json|PlanningAgent:LLM"
]

def _csv_exists(filepath: str = CSV_LOG_FILE) -> None:
    """Create the CSV file with headers if it does not exist."""
    if not os.path.exists(filepath):
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

def log_query(
    session_id: str,
    query: str,
    response: str,
    agents_invoked: list[str],
    agent_sources: dict[str, list[str]] | None = None,
    filepath: str = CSV_LOG_FILE,
) -> None:
    """
    Appends rows to the CSV log file with details of the query, response, agents invoked, and sources/tools used.
    """

    _csv_exists(filepath)

    agent_sources = agent_sources or {}
 
    # Build pipe-separated agents string
    agents_str = "|".join(agents_invoked) if agents_invoked else "Orchestrator(LLM)"

    # Build detailed sources string: "AgentName:source1,source2 | AgentName2:source1"
    source_parts: list[str] = []
    for agent in agents_invoked:
        sources = agent_sources.get(agent)
        if sources:
            source_parts.append(f"{agent}:{','.join(sources)}")
        else:
            source_parts.append(f"{agent}:LLM")
 
    if not source_parts:
        source_parts = ["Orchestrator:LLM"]
 
    sources_str = "|".join(source_parts)
 
    row = {
        "session_id":       session_id,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
        "query":            query,
        "response":         response,
        "agents_invoked":   agents_str,
        "sources_and_tools": sources_str,
    }
 
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
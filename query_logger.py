"""
Creates CSV log of queries, execution plan steps, agents invoked, and sources used.
Automatically migrates existing CSV files to include any missing columns.
"""

import csv
import os
from datetime import datetime

CSV_LOG_FILE = "query_log.csv"

CSV_COLUMNS = [
    "conversation_num",
    "session_id",
    "timestamp",
    "response_time_sec",
    "query",
    "response",
    "plan_steps",           # pipe-separated ordered steps, e.g. "Step 1: Fetch matching courses|Step 2: Rank and select top courses"
    "agents_invoked",       # pipe-separated list of agent names, e.g. "parser|data|planning|orchestrator"
    "sources_and_tools",    # structured detail of what each agent used, e.g. "data:rutgers_courses.json|planning:LLM"
    "input_tokens",
    "output_tokens",
    "satisfied",
    "feedback",
]


def _migrate_csv(filepath: str) -> None:
    """
    If the CSV exists but is missing columns (e.g. plan_steps was added later),
    rewrite it with all current columns, filling missing fields with empty strings.
    """
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_columns = reader.fieldnames or []
        missing = [col for col in CSV_COLUMNS if col not in existing_columns]

        if not missing:
            return  # Nothing to migrate

        rows = list(reader)

    for row in rows:
        for col in missing:
            row[col] = ""

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[query_logger] Migrated '{filepath}': added column(s) {missing}")


def _csv_exists(filepath: str = CSV_LOG_FILE) -> None:
    """Create the CSV file with headers if it does not exist, or migrate it if columns are missing."""
    if not os.path.exists(filepath):
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
    else:
        _migrate_csv(filepath)


def log_query(
    conversation_num: int,
    session_id: str,
    query: str,
    response: str,
    agents_invoked: list[str],
    agent_sources: dict[str, list[str]] | None = None,
    plan_steps: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    response_time_sec: float = 0.0,
    satisfied: str = "",
    feedback: str = "",
    filepath: str = CSV_LOG_FILE,
) -> None:
    """
    Appends a row to the CSV log with details of the query, execution plan,
    agents invoked, and sources/tools used.

    Args:
        session_id:     Stable ID for the current conversation session.
        query:          The raw user query string.
        response:       The final response delivered to the user.
        agents_invoked: Ordered list of agent IDs that ran this turn.
        agent_sources:  Maps agent ID → list of sources/tools it used.
        plan_steps:     Pipe-separated plan step labels from PlanningTracer.as_log_string().
        filepath:       Path to the CSV log file.
    """
    _csv_exists(filepath)

    agent_sources = agent_sources or {}

    # Pipe-separated agent names
    agents_str = "|".join(agents_invoked) if agents_invoked else "orchestrator"

    # Detailed sources string: "agent:source1,source2|agent2:source1"
    source_parts: list[str] = []
    for agent in agents_invoked:
        sources = agent_sources.get(agent)
        if sources:
            source_parts.append(f"{agent}:{','.join(sources)}")
        else:
            source_parts.append(f"{agent}:LLM")

    sources_str = "|".join(source_parts) if source_parts else "orchestrator:LLM"

    row = {
        "session_id":        session_id,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
        "response_time_sec": f"{response_time_sec:.2f}",
        "query":             query,
        "response":          response,
        "plan_steps":        plan_steps,
        "agents_invoked":    agents_str,
        "sources_and_tools": sources_str,
        "input_tokens":      input_tokens,
        "output_tokens":     output_tokens,
        "satisfied":         satisfied,
        "feedback":          feedback,
    }

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
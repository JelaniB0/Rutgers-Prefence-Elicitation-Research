"""
Creates CSV log of queries, execution plan steps, agents invoked, and sources used.
Only migrates existing CSV files when schema has actually changed.
"""

import csv
import os
from datetime import datetime

CSV_LOG_FILE = "query_log.csv"

CSV_COLUMNS = [
    "session_id",
    "timestamp",
    "response_time_sec",
    "query",
    "response",
    "plan_steps",
    "agents_invoked",
    "sources_and_tools",
    "input_tokens",
    "output_tokens",
    "satisfied",
    "feedback",
]


def _migrate_csv(filepath: str) -> None:
    """
    Rewrites CSV to match current schema exactly.
    Drops blank rows during migration.
    """
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cleaned_rows = []
    for row in rows:
        if any(v.strip() for v in row.values()):
            cleaned_rows.append({col: row.get(col, "") for col in CSV_COLUMNS})

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(cleaned_rows)


def _csv_exists(filepath: str = CSV_LOG_FILE) -> None:
    """
    Creates CSV with headers if missing.
    Only migrates if the schema has actually changed — never rewrites unnecessarily.
    """
    if not os.path.exists(filepath):
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator='\n')
            writer.writeheader()
        return

    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing_columns = next(csv.reader([line]))
                break
        else:
            existing_columns = []

    if existing_columns != CSV_COLUMNS:
        _migrate_csv(filepath)


def log_query(
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
    Appends a row to the CSV log with details of the query,
    execution plan, agents invoked, and sources/tools used.
    """
    if not query or not query.strip():
        return

    _csv_exists(filepath)

    agent_sources = agent_sources or {}

    agents_str = "|".join(agents_invoked) if agents_invoked else "orchestrator"

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
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator='\n')
        writer.writerow(row)
"""
single_agent_baseline.py

Drop-in single-agent baseline for head-to-head comparison against the
multi-agent pipeline (parser → orchestrator → data → constraint → planning).

Same I/O contract:
  - Reads rutgers_courses.json
  - Accepts optional transcript_data (same dict schema as ConversationState)
  - Returns a plain-text advisor response
  - Logs to a CSV with the same columns as query_log3

Usage:
    python single_agent_baseline.py
"""

import os
import json
import asyncio
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

from agent_framework.openai import OpenAIChatClient
from agents2.transcript_agent import TranscriptAgent
from agents2.shared_types import ConversationState

load_dotenv()


# ---------------------------------------------------------------------------
# Minimal shared types (mirrors shared_types.py, no framework dependency)
# ---------------------------------------------------------------------------

@dataclass
class SingleAgentState:
    transcript_data: dict = field(default_factory=dict)
    conversation_history: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def add_usage(self, inp: int, out: int):
        self.input_tokens += inp
        self.output_tokens += out

    def reset_usage(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})


# ---------------------------------------------------------------------------
# Course loader  (identical filter to DataAgent)
# ---------------------------------------------------------------------------

def load_courses(courses_file: str = "rutgers_courses.json") -> list[dict]:
    with open(courses_file, "r") as f:
        return json.load(f)


def _extract_prereqs(course: dict) -> str:
    """
    Prefer the structured prerequisites list.
    Fall back to the raw description string when the list is empty —
    most electives store prereq info only in the description text.
    """
    import re
    structured = course.get("prerequisites", [])
    if structured:
        return ", ".join(structured)
    desc = course.get("description", "")
    m = re.search(r'Pre(?:requisites?|req)[:\s]+([^.]+)\.', desc, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "None listed"


def courses_to_prompt_block(courses: list[dict], max_courses: int = 40) -> str:
    """Serialise the course catalogue into a compact text block."""
    lines = []
    for c in courses[:max_courses]:
        prereqs = _extract_prereqs(c)
        lines.append(
            f"- {c['code']} | {c['title']} | Credits: {c.get('credits','?')} | "
            f"Prereqs: {prereqs}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transcript summariser  (mirrors TranscriptAgent.summarize_for_prompt)
# ---------------------------------------------------------------------------

def summarize_transcript(transcript_data: dict) -> str:
    if not transcript_data:
        return ""
    completed = transcript_data.get("completed_courses", [])
    in_progress = transcript_data.get("in_progress_courses", [])
    RELEVANT_PREFIXES = (":198:", ":640:", ":960:")
    cs_done = [c for c in completed if any(p in c.get("code", "") for p in RELEVANT_PREFIXES)]
    cs_wip  = [c for c in in_progress if any(p in c.get("code", "") for p in RELEVANT_PREFIXES)]

    done_str = "\n".join(
        f"  - {c['code']}: {c['title']} ({c.get('grade', 'P')})" for c in cs_done
    ) or "  - None"
    wip_str = "\n".join(
        f"  - {c['code']}: {c['title']}" for c in cs_wip
    ) or "  - None"

    return (
        f"Year: {transcript_data.get('year_standing', 'Unknown')}\n"
        f"GPA: {transcript_data.get('cumulative_gpa', 'Unknown')}\n"
        f"Credits completed: {transcript_data.get('total_degree_credits', '?')}\n\n"
        f"CS courses completed:\n{done_str}\n\n"
        f"CS courses in progress:\n{wip_str}"
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SINGLE_AGENT_SYSTEM = """\
You are a warm, knowledgeable academic advisor for Rutgers CS students.
You have direct access to:
  1. The full Rutgers CS course catalogue (provided in each message).
  2. The student's transcript (when uploaded).

Your job is to answer course-advising questions in a single, complete response.

## Core rules
- Only reference courses explicitly present in the provided catalogue.
  Never invent course names, codes, or prerequisites.
- Only confirm prerequisites as met if they appear in the student's completed
  courses. Never infer satisfaction from similar or higher-level courses.
- When transcript data is present, always reference the student's actual
  completed courses and standing — never ask them to provide what you already have.
- If a course's prerequisite list is empty, state "None listed."
- NEVER recommend a course the student has already completed or is currently enrolled in.
  Completed and in-progress courses are for context only — not recommendations.
- NEVER include a course in the main ranked list if the student does not meet its prerequisites.
  Unmet-prereq courses belong only in the "Courses to look forward to:" section.

## Response format for course recommendations
- Recommend at most 5 courses. Never exceed this.

1) Course Name (Course Code)
   Brief description.
   Prerequisites: X, Y, Z (met / not met based on transcript).

(blank line between courses)

End with a brief "Courses to look forward to:" section for eligible-but-blocked courses.

## Tone
Warm, encouraging, specific. Give a clear ranking rationale.
"""


# ---------------------------------------------------------------------------
# The single agent
# ---------------------------------------------------------------------------

class SingleAgentAdvisor:
    """
    One LLM call per turn.  Packs catalogue + transcript + conversation
    history into context and returns a plain-text response.
    """

    def __init__(self, client: OpenAIChatClient, courses_file: str = "rutgers_courses.json",
                 model_id: str = "gpt-4.1"):
        self.client = client
        self.courses = load_courses(courses_file)
        self.agent = client.as_agent(
            instructions=SINGLE_AGENT_SYSTEM,
            name="SingleAdvisor",
        )
        self.thread = self.agent.get_new_thread()
        # Reuse the same TranscriptAgent as the pipeline — identical PDF parsing
        self._transcript_agent = TranscriptAgent(client=client, model=model_id)
        # A ConversationState is only needed as a carrier for TranscriptAgent's API
        self._transcript_state = ConversationState()

    async def load_transcript(self, file_path: str, state: SingleAgentState) -> str:
        """
        Parse a transcript PDF and populate state.transcript_data.
        Returns a human-readable confirmation string (same content the
        multi-agent pipeline prints after TranscriptExecutor runs).
        """
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        response = await self._transcript_agent.parse_transcript(
            file_path, self._transcript_state
        )

        if hasattr(response, "metadata") and response.metadata:
            state.add_usage(
                response.metadata.get("input_token_count", 0) or 0,
                response.metadata.get("output_token_count", 0) or 0,
            )

        if not response.success:
            return f"Could not parse transcript: {', '.join(response.errors)}"

        # Persist into our state (same dict schema ConversationState uses)
        state.transcript_data = self._transcript_state.transcript_data = response.data

        data = response.data
        cs_done = [c for c in data.get("completed_courses", []) if ":198:" in c.get("code", "")]
        cs_wip  = [c for c in data.get("in_progress_courses", []) if ":198:" in c.get("code", "")]
        done_str = "\n".join(f"  - {c['code']}: {c['title']} ({c.get('grade','P')})" for c in cs_done) or "  - None"
        wip_str  = "\n".join(f"  - {c['code']}: {c['title']}" for c in cs_wip) or "  - None"

        return (
            f"Transcript loaded.\n\n"
            f"Year: {data.get('year_standing')}\n"
            f"GPA: {data.get('cumulative_gpa')}\n"
            f"Credits: {data.get('total_degree_credits')}\n\n"
            f"CS courses completed:\n{done_str}\n\n"
            f"CS courses in progress:\n{wip_str}"
        )

    def _build_prompt(self, user_query: str, state: SingleAgentState) -> str:
        catalogue_block = courses_to_prompt_block(self.courses)
        transcript_block = summarize_transcript(state.transcript_data)

        parts = [f"## Student query\n{user_query}\n"]

        if transcript_block:
            parts.append(
                f"## Student transcript\n"
                f"Transcript is on file — use it. Do not ask for it.\n\n"
                f"{transcript_block}\n"
            )
        else:
            parts.append("## Student transcript\nNo transcript uploaded.\n")

        parts.append(f"## Course catalogue\n{catalogue_block}\n")

        return "\n".join(parts)

    async def answer(self, user_query: str, state: SingleAgentState) -> str:
        prompt = self._build_prompt(user_query, state)
        raw = await self.agent.run(prompt, thread=self.thread)

        if hasattr(raw, "usage_details") and raw.usage_details:
            state.add_usage(
                raw.usage_details.get("input_token_count", 0) or 0,
                raw.usage_details.get("output_token_count", 0) or 0,
            )

        return raw.content if hasattr(raw, "content") else str(raw)


# ---------------------------------------------------------------------------
# CSV logger  (same columns as query_log3)
# ---------------------------------------------------------------------------

LOG_FILE = "single_agent2.csv"

HEADER = (
    "session_id,timestamp,response_time_sec,query,response,"
    "plan_steps,agents_invoked,sources_and_tools,"
    "input_tokens,output_tokens,"
    "satisfied,feedback,hallucinated,hallucination_type,hallucination_notes\n"
)


def _escape(s: str) -> str:
    """Minimal CSV escaping."""
    s = s.replace('"', '""')
    if any(c in s for c in (',', '"', '\n')):
        s = f'"{s}"'
    return s


def log_query(
    session_id: str, query: str, response: str,
    input_tokens: int, output_tokens: int,
    response_time_sec: float,
    satisfied: str, feedback: str,
    hallucinated: str, hallucination_type: str, hallucination_notes: str,
):
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write(HEADER)

    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    row = ",".join([
        _escape(session_id),
        _escape(timestamp),
        f"{response_time_sec:.2f}",
        _escape(query),
        _escape(response),
        "single_agent",               # plan_steps
        "single_agent",               # agents_invoked
        "single_agent:LLM,rutgers_courses.json",  # sources_and_tools
        str(input_tokens),
        str(output_tokens),
        _escape(satisfied),
        _escape(feedback),
        _escape(hallucinated),
        _escape(hallucination_type),
        _escape(hallucination_notes),
    ])
    with open(LOG_FILE, "a") as f:
        f.write(row + "\n")


# ---------------------------------------------------------------------------
# Feedback collection  (identical to main.py)
# ---------------------------------------------------------------------------

def collect_feedback() -> tuple[str, str, str, str, str]:
    satisfied = feedback = hallucinated = hallucination_type = hallucination_notes = "NULL"
    try:
        raw = input("Satisfied? (y/n, Enter to skip): ").strip().lower()
        if raw in ("y", "yes"):
            satisfied = "yes"
            feedback = input("Feedback? (Enter to skip): ").strip() or "NULL"
        elif raw in ("n", "no"):
            satisfied = "no"
            feedback = input("Feedback? (Enter to skip): ").strip() or "NULL"
    except (KeyboardInterrupt, EOFError):
        return satisfied, feedback, hallucinated, hallucination_type, hallucination_notes

    try:
        h = input("Hallucination? (y/n, Enter to skip): ").strip().lower()
        if h in ("y", "yes"):
            hallucinated = "yes"
            print("Type: 1=course_invented 2=prereq_wrong 3=code_wrong 4=ranking_wrong 5=other")
            t = input("Number (Enter to skip): ").strip()
            hallucination_type = {
                "1": "course_invented", "2": "prereq_wrong",
                "3": "code_wrong", "4": "ranking_wrong", "5": "other",
            }.get(t, "other") if t else ""
            hallucination_notes = input("Notes (Enter to skip): ").strip()
        elif h in ("n", "no"):
            hallucinated = "no"
    except (KeyboardInterrupt, EOFError):
        pass

    return satisfied, feedback, hallucinated, hallucination_type, hallucination_notes


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

async def main():
    print("Rutgers CS Course Advisor — Single Agent Baseline")
    print("(same eval protocol as multi-agent; logs to single_agent2.csv)")
    print("Type 'quit' to exit.\n")

    client = OpenAIChatClient(
        base_url=os.environ.get("GITHUB_ENDPOINT"),
        api_key=os.environ.get("GITHUB_TOKEN"),
        model_id=os.environ.get("GITHUB_MODEL_ID"),
    )

    model_id = os.environ.get("GITHUB_MODEL_ID")
    advisor = SingleAgentAdvisor(client, model_id=model_id)
    state = SingleAgentState()
    session_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        # Transcript upload — same PDF path detection as main.py
        pdf_match = re.search(r'[\w./ \\-]+\.pdf', user_input, re.IGNORECASE)
        if pdf_match:
            file_path = pdf_match.group().strip()
            print(f"Loading transcript: {file_path} ...")
            state.reset_usage()
            confirmation = await advisor.load_transcript(file_path, state)
            print(f"\nAdvisor: {confirmation}\n")
            state.add_message("user", user_input)
            state.add_message("assistant", confirmation)
            continue
        state.reset_usage()
        state.add_message("user", user_input)
        turn_start = datetime.now()

        try:
            response = await advisor.answer(user_input, state)
        except Exception as e:
            response_time = (datetime.now() - turn_start).total_seconds()
            err_str = str(e)
            if "tokens_limit_reached" in err_str or "413" in err_str:
                print("\n[Context limit reached] The conversation history is too long for this model. Starting a new session is recommended.\n")
                log_query(
                    session_id=session_id,
                    query=user_input,
                    response="ERROR: tokens_limit_reached",
                    input_tokens=state.input_tokens,
                    output_tokens=state.output_tokens,
                    response_time_sec=response_time,
                    satisfied="no",
                    feedback="auto-logged: token limit crash",
                    hallucinated="NULL",
                    hallucination_type="",
                    hallucination_notes="",
                )
            else:
                print(f"[Error] {e}")
                import traceback; traceback.print_exc()
            continue

        response_time = (datetime.now() - turn_start).total_seconds()
        state.add_message("assistant", response)
        print(f"\nAdvisor: {response}\n")

        satisfied, feedback, hallucinated, h_type, h_notes = collect_feedback()

        log_query(
            session_id=session_id,
            query=user_input,
            response=response,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            response_time_sec=response_time,
            satisfied=satisfied,
            feedback=feedback,
            hallucinated=hallucinated,
            hallucination_type=h_type,
            hallucination_notes=h_notes,
        )


if __name__ == "__main__":
    asyncio.run(main())
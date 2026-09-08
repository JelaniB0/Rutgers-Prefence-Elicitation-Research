"""
single_agent.py

Single-agent baseline for benchmarking against the multi-agent hub-and-spoke framework.

Architecture:
  - One LLM call per turn (after pre-fetching external data in Python)
  - Same data sources as multi-agent: rutgers_courses.json, Rutgers SOC API, transcript PDF
  - ChromaDB + semantic search for course retrieval (same as DataAgent)
  - No routing, no orchestration, no inter-agent messaging
  - Conversation history maintained across turns

Benchmark metrics tracked per turn:
  - input_tokens, output_tokens
  - wall-clock latency (seconds)
  - response text (for manual accuracy eval)

Run:
    python single_agent.py
"""

import os
import re
import json
import time
import asyncio
import pdfplumber
import httpx
import chromadb

from datetime import date, datetime
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COURSES_FILE   = "rutgers_courses.json"
CHROMA_PATH    = "./chroma_db_single"   # separate DB so we don't collide with multi-agent
RUTGERS_SOC    = "https://classes.rutgers.edu/soc/api/courses.json"
EMBED_MODEL    = "text-embedding-3-small"
CHAT_MODEL     = os.environ.get("GITHUB_MODEL_ID", "gpt-4.1")
GITHUB_TOKEN   = os.environ.get("GITHUB_TOKEN")
GITHUB_BASE    = os.environ.get("GITHUB_ENDPOINT", "https://models.inference.ai.azure.com/")
MAX_RESULTS    = 5        # courses to surface per recommendation turn
RETRIEVE_K     = 20       # candidates pulled from vector DB before LLM filter
CACHE_TTL      = 60 * 60 * 12  # 12 h SOC cache

# ---------------------------------------------------------------------------
# System prompt — collapses orchestrator + parser + constraint + planning roles
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a warm, knowledgeable academic advisor for Rutgers Computer Science students.

In each turn you receive:
  - The student's message
  - A course catalog section (semantically retrieved courses relevant to their query)
  - Live semester availability for those courses (from the Rutgers Schedule of Classes)
  - Optionally: the student's parsed transcript (completed courses, GPA, credits, standing)
  - Conversation history for context

YOUR RESPONSIBILITIES IN ONE TURN:
1. Identify the student's intent:
     course_recommendation  — they want suggestions
     course_info            — they ask about a specific named course
     prerequisite_check     — "can I take X", "do I need Y before Z"
     general_question       — program/graduation/policy question
     transcript_upload      — they want to share their transcript
     off_topic              — nothing to do with CS advising

2. For course_recommendation:
   a. Select the most relevant courses from the catalog provided.
   b. If a transcript is present, filter out completed/in-progress courses.
   c. Check prerequisites: a course is ELIGIBLE if all its prereqs appear in
      the student's completed OR in-progress courses.
      - AND logic: ALL listed prereqs must be met.
      - OR logic: ANY ONE is sufficient.
      - "Permission of instructor" -> treat as met.
      - No prereqs listed -> always eligible.
   d. Credit-standing guidance (soft, never hard-block):
      - 100/200-level: open to all
      - 300-level: typically Junior standing (60+ credits)
      - 400-level: typically Senior standing (90+ credits)
      If the student is below the typical standing, mention it but still
      include the course — rank it lower.
   e. Rank and present the top {max_results} courses.

3. For prerequisite_check:
   State clearly whether the student meets prerequisites, what is missing,
   and what they should take first to become eligible.

4. For course_info:
   Describe the course: content, prerequisites, typical audience, credits.

5. RESPONSE FORMAT for recommendations (plain text, no markdown fences):

   1) Course Title (Course Code) — Offered: Yes/No/Unverified
      Brief description of what the course covers and why it fits this student.
      Prerequisites: list them, or "None".

   2) Next Course ...

   Leave a blank line between courses.
   End with a short paragraph summarising the plan.

6. HARD RULES:
   - Only reference courses present in the catalog injected below.
     Never invent or recall courses from memory.
   - If a course is NOT offered this semester, say so and explain whether
     it is still worth planning for.
   - If no transcript is provided, do not guess the student's completed
     courses. Note prerequisites but do not block any course.
   - Never output raw JSON or code blocks in your response.
   - Keep the tone encouraging and advisor-like.
   - For follow-up questions referencing previous recommendations (e.g. "which of those",
     "prioritize them", "take asap"), answer directly from the conversation history.
     Do not ignore prior context.
   - If the student's query is too vague to retrieve relevant courses 
    (e.g. no subject, topic, interest, or course name mentioned at all), 
    ask ONE short friendly clarifying question instead of guessing.
    Otherwise always attempt to answer.
  - If off topic, dont answer the question and politely steer them back to CS advising topics. 
""".format(max_results=MAX_RESULTS)

# Helpers — course loading, embedding, vector DB

def load_courses(path: str) -> list[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("courses", [])
    except Exception as e:
        print(f"[SingleAgent] Could not load courses: {e}")
        return []


def extract_prereqs(description: str) -> tuple[str, str]:
    """Split prereq sentence from rest of description (mirrors DataAgent)."""
    match = re.match(r'(Prerequisites?:.*?\.)\s*(.*)', description, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", description


def enrich_course(course: dict, code_to_title: dict) -> dict:
    course = course.copy()
    prereq_text, clean_desc = extract_prereqs(course.get("description", ""))
    
    # fallback to list field if description parsing found nothing
    if not prereq_text and course.get("prerequisites"):
        prereqs = course["prerequisites"]
        if isinstance(prereqs, list):
            resolved = [f"{p} ({code_to_title.get(p, p)})" for p in prereqs]
            prereq_text = "Prerequisites: " + ", ".join(resolved)
    
    for code, title in code_to_title.items():
        prereq_text = prereq_text.replace(code, f"{code} ({title})")
    course["prerequisites"] = prereq_text
    course["description"]   = clean_desc
    return course

def course_to_document(course: dict) -> str:
    parts = [
        f"Course: {course.get('title', '')}",
        f"Code: {course.get('code', '')}",
        f"Description: {course.get('description', '')}",
    ]
    if course.get("prerequisites"):
        parts.append(f"Prerequisites: {course['prerequisites']}")
    if course.get("topics"):
        parts.append(f"Topics: {', '.join(course['topics'])}")
    return " ".join(parts)


def resolve_semester(query: str) -> dict:
    """Mirrors DataAgent.resolve_semester exactly."""
    q     = query.lower()
    today = date.today()
    m, y  = today.month, today.year

    year_match    = re.search(r"20\d{2}", q)
    explicit_year = int(year_match.group()) if year_match else None

    if "spring" in q:
        term, yr = 1, explicit_year or (y + 1 if m >= 9 else y)
    elif "summer" in q:
        term, yr = 7, explicit_year or y
    elif "winter" in q:
        term, yr = 0, explicit_year or (y + 1 if m >= 9 else y)
    elif "fall" in q or "autumn" in q:
        term, yr = 9, explicit_year or (y + 1 if m >= 9 else y)
    elif "next semester" in q or "next sem" in q:
        if m <= 5:    term, yr = 9, y
        elif m <= 8:  term, yr = 9, y
        elif m <= 11: term, yr = 1, y + 1
        else:         term, yr = 1, y + 1
    elif "this semester" in q or "current" in q:
        if m <= 1:    term, yr = 0, y
        elif m <= 5:  term, yr = 1, y
        elif m <= 8:  term, yr = 7, y
        else:         term, yr = 9, y
    else:
        if m <= 1:    term, yr = 0, y
        elif m <= 5:  term, yr = 1, y
        elif m <= 8:  term, yr = 7, y
        else:         term, yr = 9, y

    return {"term": term, "year": yr}


def extract_transcript_text(pdf_path: str) -> str:
    """Two-column extraction, mirrors TranscriptAgent._extract_text."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                w, h = page.width, page.height
                for col in [page.crop((0, 0, w / 2, h)), page.crop((w / 2, 0, w, h))]:
                    t = col.extract_text()
                    if t:
                        text += t + "\n"
    except Exception as e:
        raise ValueError(f"PDF read failed: {e}")
    return text


# SingleAgent class

class SingleAgent:
    """
    Stateful single-agent advisor. One LLM call per user turn.

    Pre-turn (Python, not LLM):
      1. Resolve semester from query
      2. Fetch live SOC data (cached)
      3. Semantic search -> top-K candidate courses
      4. LLM filter call to prune weak matches (1 cheap call, same as DataAgent)
      5. Build enriched context string

    Main turn (LLM):
      6. Single chat completion with full context injected
    """

    def __init__(self):
        # Async OpenAI client (GitHub Models endpoint)
        self.client = AsyncOpenAI(
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_BASE,
        )

        # Course data
        self.courses       = load_courses(COURSES_FILE)
        self.code_to_title = {c["code"]: c["title"] for c in self.courses}
        self.courses       = [enrich_course(c, self.code_to_title) for c in self.courses]
        self.stated_completed: set[str] = set()

        # Vector DB (separate collection from multi-agent)
        embed_fn = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
            api_key=GITHUB_TOKEN,
            api_base=GITHUB_BASE,
            model_name=EMBED_MODEL,
        )
        chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            self.collection = chroma.get_collection("single_agent_courses", embedding_function=embed_fn)
        except Exception:
            self.collection = chroma.create_collection(
                "single_agent_courses",
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )

        self._index_if_empty()

        # SOC cache  {term_year: {"courses": set, "fetched_at": float}}
        self.soc_cache: dict = {}

        # Conversation history  [{"role": "user"|"assistant", "content": str}]
        self.history: list[dict] = []

        # Transcript data (populated after parse_transcript call)
        self.transcript: Optional[dict] = None

        # Cumulative token usage across session
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

        print(f"[SingleAgent] Loaded {len(self.courses)} courses")

    # Vector DB indexing

    def _index_if_empty(self):
        if self.collection.count() > 0:
            return
        print("[SingleAgent] Indexing courses into vector DB...")
        docs, metas, ids = [], [], []
        for course in self.courses:
            docs.append(course_to_document(course))
            metas.append({
                "code":  course.get("code", ""),
                "title": course.get("title", ""),
            })
            ids.append(course.get("code", f"course_{len(ids)}"))
        self.collection.add(documents=docs, metadatas=metas, ids=ids)
        print(f"[SingleAgent] Indexed {len(docs)} courses")

    # SOC API

    async def _fetch_offered(self, semester: dict) -> Optional[set[str]]:
        """Returns set of offered course numbers (e.g. '314') or None on failure."""
        key = f"{semester['term']}_{semester['year']}"
        now = time.time()
        cached = self.soc_cache.get(key)
        if cached and (now - cached["fetched_at"]) < CACHE_TTL:
            return cached["courses"]
        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.get(RUTGERS_SOC, params={
                    "year":   semester["year"],
                    "term":   semester["term"],
                    "campus": "NB",
                })
                resp.raise_for_status()
                offered = {
                    str(c["courseNumber"])
                    for c in resp.json()
                    if str(c.get("subject", "")) == "198" and c.get("level") == "U"
                }
            self.soc_cache[key] = {"courses": offered, "fetched_at": now}
            print(f"[SingleAgent] SOC: {len(offered)} CS courses offered for {key}")
            return offered
        except Exception as e:
            print(f"[SingleAgent] SOC unavailable: {e}")
            return None

    # Transcript parsing

    async def parse_transcript(self, pdf_path: str) -> str:
        """Parse transcript PDF, store result, return human-readable summary."""
        raw_text = extract_transcript_text(pdf_path)
        if not raw_text.strip():
            return "I couldn't extract any text from that PDF."

        transcript_system = """\
You are a transcript parser for Rutgers University.
Extract academic data and return ONLY valid JSON — no markdown, no explanation.

Course code format: "SCH:DEPT:NUM" e.g. "01:198:111"

Return exactly:
{
  "student_name": "FIRST LAST",
  "student_id": "123456789",
  "cumulative_gpa": 3.74,
  "total_degree_credits": 91.0,
  "year_standing": "Freshman|Sophomore|Junior|Senior",
  "completed_courses":  [{"code":"","title":"","credits":0.0,"grade":"","semester":""}],
  "in_progress_courses":[{"code":"","title":"","credits":0.0,"semester":""}],
  "ap_credits":         [{"code":"","title":"","credits":0.0}],
  "transfer_courses":   [{"code":"","title":"","credits":0.0}]
}

Rules:
- cumulative_gpa: use the LAST cumulative avg listed
- total_degree_credits: use the LAST degree credits earned value
- completed_courses: only courses with a letter grade or PA/P
- in_progress_courses: current semester, no grade yet
- year_standing: infer from credits (<30 Freshman, <60 Sophomore, <90 Junior, 90+ Senior)
- ignore 0-credit duplicate lab lines
"""
        resp = await self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": transcript_system},
                {"role": "user",   "content": raw_text},
            ],
            max_tokens=2000,
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return "I couldn't parse that transcript. Please make sure it's a Rutgers PDF transcript."
        try:
            self.transcript = json.loads(match.group())
        except json.JSONDecodeError:
            return "Transcript JSON was malformed. Please try again."

        self.total_input_tokens  += resp.usage.prompt_tokens
        self.total_output_tokens += resp.usage.completion_tokens

        data    = self.transcript
        cs_done = [c for c in data.get("completed_courses",   []) if ":198:" in c.get("code", "")]
        cs_wip  = [c for c in data.get("in_progress_courses", []) if ":198:" in c.get("code", "")]

        done_str = "\n".join(f"  - {c['code']}: {c['title']} ({c.get('grade','P')})" for c in cs_done) or "  - None"
        wip_str  = "\n".join(f"  - {c['code']}: {c['title']}"                       for c in cs_wip)  or "  - None"

        return (
            f"Got it! Transcript parsed.\n\n"
            f"Year: {data.get('year_standing')}  |  GPA: {data.get('cumulative_gpa')}  "
            f"|  Credits: {data.get('total_degree_credits')}\n\n"
            f"CS Completed:\n{done_str}\n\n"
            f"CS In Progress:\n{wip_str}\n\n"
            f"I'll factor all of this into future recommendations."
        )

    # Core retrieval — mirrors DataAgent._rag_retrieve

    async def _retrieve_courses(self, user_query: str, semester: dict, intent: str = None) -> list[dict]:
        """
        1. Rewrite query for semantic search (1 cheap LLM call)
        2. Vector search -> top-K candidates
        3. Filter by SOC availability
        4. Filter out completed/in-progress if transcript present
        5. LLM filter to prune weak matches (1 cheap LLM call)
        Returns enriched course list.
        """

        if intent in ("course_info", "prerequisite_check"):
            code_match = re.search(r'(?:198[:\s])?(\d{3})', user_query)
            if code_match:
                num = code_match.group(1)
                direct = [c for c in self.courses if c.get("code", "").split(":")[-1].strip() == num]
                if direct:
                    return direct
                
        # query rewrite
        try:
            rw_resp = await self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rewrite the following into a strong 1-2 sentence semantic search query "
                        f"for a university course catalog. Focus on topics, skills, and subject matter.\n\n"
                        f"Original query: {user_query}"
                    ),
                }],
                max_tokens=100,
                temperature=0,
            )
            search_query = rw_resp.choices[0].message.content or user_query
            self.total_input_tokens  += rw_resp.usage.prompt_tokens
            self.total_output_tokens += rw_resp.usage.completion_tokens
        except Exception:
            search_query = user_query

        # vector search
        results   = self.collection.query(query_texts=[search_query], n_results=RETRIEVE_K)
        codes     = results["ids"][0]
        distances = results["distances"][0]

        candidates = []
        for code, dist in zip(codes, distances):
            course = next((c for c in self.courses if c.get("code") == code), None)
            if course:
                c = course.copy()
                c["semantic_similarity"] = round(1 - dist, 4)
                candidates.append(c)

        # SOC filter
        offered = await self._fetch_offered(semester)
        if offered:
            candidates = [
                c for c in candidates
                if c.get("code", "").split(":")[-1].strip() in offered
            ]

        # transcript filter
        if self.transcript:
            completed   = {c["code"] for c in self.transcript.get("completed_courses",   [])}
            in_progress = {c["code"] for c in self.transcript.get("in_progress_courses", [])}
            transfer    = {c["code"] for c in self.transcript.get("transfer_courses",     [])}
            ap          = {c["code"] for c in self.transcript.get("ap_credits",           [])}
            exclude     = completed | in_progress | transfer | ap | self.stated_completed
            candidates  = [c for c in candidates if c.get("code") not in exclude]

        # LLM relevance filter
        if candidates:
            try:
                filter_resp = await self.client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"From the courses below, keep only those relevant to the student's query. "
                            f"Be generous — keep at least 8 if available. "
                            f"Only remove courses that are clearly unrelated.\n\n"
                            f"Student query: {user_query}\n\n"
                            f"Courses:\n"
                            + json.dumps([
                                {"code": c.get("code"), "title": c.get("title"),
                                 "description": (c.get("description") or "")[:200]}
                                for c in candidates
                            ], indent=2)
                            + '\n\nReturn ONLY JSON: {"keep": ["01:198:111", ...]}'
                        ),
                    }],
                    max_tokens=300,
                    temperature=0,
                )
                raw_filter = filter_resp.choices[0].message.content or ""
                self.total_input_tokens  += filter_resp.usage.prompt_tokens
                self.total_output_tokens += filter_resp.usage.completion_tokens
                m = re.search(r"\{.*\}", raw_filter, re.DOTALL)
                if m:
                    keep_codes = json.loads(m.group()).get("keep", [])
                    if keep_codes:
                        candidates = [c for c in candidates if c.get("code") in keep_codes]
            except Exception as e:
                print(f"[SingleAgent] LLM filter failed, keeping all: {e}")

        return candidates[:MAX_RESULTS + 5]  # slight buffer for the main LLM to choose from


    # Context builders

    def _build_course_context(self, courses: list[dict], offered: Optional[set], semester: dict) -> str:
        """Build course catalog string injected into the prompt."""
        if not courses:
            return "=== COURSE CATALOG ===\nNo matching courses found for this query.\n"

        term_map  = {1: "Spring", 7: "Summer", 9: "Fall", 0: "Winter"}
        sem_label = f"{term_map.get(semester.get('term'), '?')} {semester.get('year', '?')}"
        lines     = [f"=== COURSE CATALOG (Semester: {sem_label}) ==="]

        for course in courses:
            code    = course.get("code", "")
            title   = course.get("title", "")
            desc    = (course.get("description") or "")[:300]
            prereqs = course.get("prerequisites") or "None"
            credits = course.get("credits", "3")

            num = code.split(":")[-1].strip()
            if offered is None:
                offered_str = "Unverified (SOC unavailable)"
            elif num in offered:
                offered_str = "Yes"
            else:
                offered_str = "No"

            lines.append(
                f"\n[{code}] {title} | Credits: {credits} | Offered: {offered_str}\n"
                f"  Description: {desc}\n"
                f"  Prerequisites: {prereqs}"
            )

        return "\n".join(lines)

    def _build_transcript_context(self) -> str:
        """Compact transcript summary injected into every prompt when available."""
        if not self.transcript:
            if self.stated_completed:
                return f"No transcript on file.\nStated completed courses: {', '.join(self.stated_completed)}"
            return "No transcript on file."

        td      = self.transcript
        cs_done = [c for c in td.get("completed_courses",   []) if ":198:" in c.get("code", "")]
        cs_wip  = [c for c in td.get("in_progress_courses", []) if ":198:" in c.get("code", "")]
        all_done = (
            td.get("completed_courses", []) +
            td.get("transfer_courses",  []) +
            td.get("ap_credits",        [])
        )
        all_done_str = ", ".join(f"{c['code']}({c.get('grade', 'P')})" for c in all_done) or "None"

        stated_str = (
            f"\nStated completed (no transcript): {', '.join(self.stated_completed)}"
            if self.stated_completed else ""
        )

        return (
            f"=== STUDENT TRANSCRIPT ===\n"
            f"Name:            {td.get('student_name')}\n"
            f"Year Standing:   {td.get('year_standing')}\n"
            f"Cumulative GPA:  {td.get('cumulative_gpa')}\n"
            f"Degree Credits:  {td.get('total_degree_credits')}\n"
            f"CS Completed:    {', '.join(c['code'] for c in cs_done) or 'None'}\n"
            f"CS In-Progress:  {', '.join(c['code'] for c in cs_wip) or 'None'}\n"
            f"All Completed:   {all_done_str}{stated_str}\n"
        )

    # Main turn

    async def _detect_intent(self, query: str) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Classify this query into one of: course_info, course_recommendation, "
                        f"prerequisite_check, general.\n\nQuery: {query}\n\nReturn only the intent word."
                    )
                }],
                max_tokens=10,
                temperature=0,
            )
            self.total_input_tokens  += resp.usage.prompt_tokens
            self.total_output_tokens += resp.usage.completion_tokens
            return resp.choices[0].message.content.strip().lower()
        except Exception:
            return "general"  # safe fallback

    async def chat(self, user_query: str) -> tuple[str, dict]:
        """
        Process one user turn.

        Returns:
            response_text  — advisor reply
            metrics        — {input_tokens, output_tokens, latency_sec,
                              courses_retrieved, semester}
        """
        turn_start = time.time()

        # Detect transcript upload intent cheaply (no LLM needed)
        lower = user_query.lower()
        if re.search(r'\.pdf', lower) or any(k in lower for k in
                ("transcript", "my courses", "courses i've taken", "here's my pdf")):
            path_match = re.search(r'[\w./ \\-]+\.pdf', user_query, re.IGNORECASE)
            if path_match:
                pdf_path = path_match.group().strip()
                if os.path.exists(pdf_path):
                    summary = await self.parse_transcript(pdf_path)
                    self.history.append({"role": "user",      "content": user_query})
                    self.history.append({"role": "assistant",  "content": summary})
                    return summary, {
                        "input_tokens":      0,
                        "output_tokens":     0,
                        "latency_sec":       round(time.time() - turn_start, 2),
                        "courses_retrieved": 0,
                        "semester":          {},
                    }

        # Resolve semester and retrieve courses
        intent = await self._detect_intent(user_query)
        semester = resolve_semester(user_query)
        offered  = await self._fetch_offered(semester)
        courses  = await self._retrieve_courses(user_query, semester, intent=intent)
        course_context     = self._build_course_context(courses, offered, semester)
        transcript_context = self._build_transcript_context()

        # Build messages: system + history + fresh context injection + new user turn
        context_injection = (
            f"{transcript_context}\n\n"
            f"{course_context}"
        )

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self.history[-8:]
            + [{"role": "system", "content": context_injection}]
            + [{"role": "user",   "content": user_query}]
        )

        # Single LLM call
        resp = await self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=1200,
            temperature=0.3,
        )

        response_text = resp.choices[0].message.content or "(no response)"
        in_tok  = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
        self.total_input_tokens  += in_tok
        self.total_output_tokens += out_tok

        # Update history
        self.history.append({"role": "user",      "content": user_query})
        self.history.append({"role": "assistant",  "content": response_text})

        metrics = {
            "input_tokens":      in_tok,
            "output_tokens":     out_tok,
            "latency_sec":       round(time.time() - turn_start, 2),
            "courses_retrieved": len(courses),
            "semester":          semester,
        }
        return response_text, metrics


# Benchmark logger — matches multi-agent CSV format exactly

LOG_FILE = "single_agent_log.csv"

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

def _ensure_csv_header(log_file: str):
    if not os.path.exists(log_file):
        import csv
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def log_turn(
    session_id:  str,
    query:       str,
    response:    str,
    metrics:     dict,
    satisfied:   str = "",
    feedback:    str = "",
    log_file:    str = LOG_FILE,
):
    import csv

    _ensure_csv_header(log_file)

    sem       = metrics.get("semester", {})
    term_map  = {1: "spring", 7: "summer", 9: "fall", 0: "winter"}
    sem_label = f"{term_map.get(sem.get('term'), '?')}_{sem.get('year', '?')}"
    plan_steps = f"query_rewrite -> vector_search -> soc_filter({sem_label}) -> llm_filter -> llm_respond"

    row = {
        "session_id":        session_id,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
        "response_time_sec": metrics.get("latency_sec", 0),
        "query":             query,
        "response":          response,
        "plan_steps":        plan_steps,
        "agents_invoked":    "single_agent",
        "sources_and_tools": "single_agent:LLM,rutgers_courses.json,SOC_API",
        "input_tokens":      metrics.get("input_tokens",  0),
        "output_tokens":     metrics.get("output_tokens", 0),
        "satisfied":         satisfied,
        "feedback":          feedback,
    }

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


# standalone 

async def main():
    print("=" * 60)
    print(" Rutgers CS Advisor — Single Agent Baseline")
    print("=" * 60)
    print("Ask me about courses, prerequisites, or upload your transcript.")
    print("Type 'quit' to exit.  Type 'stats' to see session token usage.\n")

    agent      = SingleAgent()
    session_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\nSession totals — Input tokens: {agent.total_input_tokens} | "
                  f"Output tokens: {agent.total_output_tokens} | "
                  f"Total: {agent.total_input_tokens + agent.total_output_tokens}")
            print("Good luck with your courses!")
            break

        if user_input.lower() == "stats":
            print(f"  Input tokens:  {agent.total_input_tokens}")
            print(f"  Output tokens: {agent.total_output_tokens}")
            print(f"  Total:         {agent.total_input_tokens + agent.total_output_tokens}")
            continue

        if not user_input:
            continue

        try:
            response, metrics = await agent.chat(user_input)
            print(f"\nAdvisor: {response}\n")
            print(f"  [tokens in:{metrics['input_tokens']} out:{metrics['output_tokens']} "
                  f"| latency:{metrics['latency_sec']}s | courses:{metrics['courses_retrieved']}]\n")

            satisfied = ""
            feedback  = ""
            try:
                raw = input("Satisfied? (y/n or Enter to skip): ").strip().lower()
                if raw in ("y", "yes"):
                    satisfied = "yes"
                    feedback  = input("Feedback (Enter to skip): ").strip()
                elif raw in ("n", "no"):
                    satisfied = "no"
                    feedback  = input("Feedback (Enter to skip): ").strip()
            except (KeyboardInterrupt, EOFError):
                pass

            log_turn(
                session_id=session_id,
                query=user_input,
                response=response,
                metrics=metrics,
                satisfied=satisfied,
                feedback=feedback,
            )

        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
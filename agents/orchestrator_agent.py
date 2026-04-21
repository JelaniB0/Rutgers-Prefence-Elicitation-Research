import json
from dataclasses import dataclass, field
from typing import Optional

from agent_framework.openai import OpenAIChatClient
from agent_framework import Executor, WorkflowContext, handler

from agents.shared_types import ConversationState

# Message Type
class UserQuery:
    def __init__(self, user_query: str, conversation_state: ConversationState):
        self.user_query = user_query
        self.conversation_state = conversation_state


class OrchestratorRequest:
    def __init__(self, user_query: str, parsed_data: dict, conversation_state: ConversationState):
        self.user_query = user_query
        self.parsed_data = parsed_data
        self.conversation_state = conversation_state


class AgentResult:
    def __init__(self, user_query: str, parsed_data: dict, agent_name: str,
                 data: dict, conversation_state: ConversationState):
        self.user_query = user_query
        self.parsed_data = parsed_data
        self.agent_name = agent_name
        self.data = data
        self.conversation_state = conversation_state

# Agents 

AGENT_REGISTRY = {
    "transcript": {
        "description": "Parses uploaded transcript. Call only on transcript_upload intent. Terminal.",
        "terminal": True,
    },
    "data_fetch": {
        "description": "Fetches courses matching filters. Call first for recommendations.",
        "terminal": False,
    },
    "data_lookup": {
        "description": "Looks up a specific course by name/ID. Use for course_info intent.",
        "terminal": False,
    },
    "data_prereq": {
        "description": "Gets prerequisites for a specific course. Use for prerequisite_check intent.",
        "terminal": False,
    },
    "constraint_full": {
        "description": "Validates course list against transcript. Requires data_fetch + transcript.",
        "terminal": False,
    },
    "constraint_prereq": {
        "description": "Checks if student meets prereqs for one course. Requires data_prereq + transcript.",
        "terminal": False,
    },
    "planning": {
        "description": "REQUIRED for all course_recommendation intents. Ranks courses. Call after constraint_full if transcript available, else after data_fetch.",
        "terminal": False,
    },
}

AGENT_REGISTRY_SUMMARY = "\n".join(
    f"- {name}: {meta['description']}"
    for name, meta in AGENT_REGISTRY.items()
)

# Orchestrator Routing context

@dataclass
class RoutingContext:
    user_query: str
    parsed_data: dict
    has_transcript: bool
    resolved_courses: dict = field(default_factory=dict)
    accumulated_results: dict[str, dict] = field(default_factory=dict)
    agents_call_order: list[str] = field(default_factory=list)  # helps log dynamic agent calls by orchestrator. 
    resolved_semester: dict = field(default_factory=dict)

    def _slim_results(self) -> dict:
        slim = {}
        for key, val in self.accumulated_results.items():
            if isinstance(val, dict) and "courses" in val:
                slim_courses = []
                for c in val["courses"][:5]:
                    actual = c.get("course", c)
                    slim_courses.append({
                        "code": actual.get("code"),
                        "title": actual.get("title"),
                        "prerequisites": actual.get("prerequisites") or actual.get("description", "")
                    })
                slim[key] = {"courses": slim_courses}
            elif key == "constraint_data":
                slim[key] = {"summary": "constraint check complete"}
            else:
                slim[key] = val
        return slim
    
    def _slim_parsed_data(self) -> dict:
        d = {k: v for k, v in self.parsed_data.items() if v not in (None, [], {}, "")}
        if "entities" in d:
            d["entities"] = {k: v for k, v in d["entities"].items() if v not in (None, [], "")}
        return d

    def to_prompt(self) -> str:
        resolved_section = ""
        if self.resolved_courses:
            term_map = {1: "Spring", 7: "Summer", 9: "Fall", 0: "Winter"}
            if isinstance(self.resolved_semester, dict):
                semester_label = f"{term_map.get(self.resolved_semester.get('term', ''), 'Unknown')} {self.resolved_semester.get('year', '')}"
            else:
                semester_label = "current semester"

            lines = "\n".join(
                f"- {v['title']} ({code}): {'Offered' if v['offered'] == True else 'Not offered' if v['offered'] == False else 'Availability unverified — must check'} for {semester_label}"
                for code, v in self.resolved_courses.items()
            )
            resolved_section = f"## Courses Already Resolved This Session\n{lines}\n"

        return f"""\
    ## Student Query
    {self.user_query}

    ## Parsed Intent & Entities
    {json.dumps(self.parsed_data, indent=2)}

    ## Context
    - Transcript on file: {self.has_transcript}

    {resolved_section}
    ## Agents Available
    {AGENT_REGISTRY_SUMMARY}

    ## Results Collected So Far
    {json.dumps(self._slim_results(), indent=2) if self.accumulated_results else "None yet."}

    ## Agents Already Called (do NOT call these again)
    {list(self.accumulated_results.keys()) if self.accumulated_results else "None"}
    """

# Routing decision

@dataclass
class RoutingDecision:
    reasoning: str
    mode: str                   # "route" | "clarify" | "respond"
    next_agents: list[str]
    response: Optional[str]

    @classmethod
    def from_llm_output(cls, raw: str) -> "RoutingDecision":
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)

        raw_agents = parsed.get("next_agents", [])
        # flatten in case LLM returns nested lists or dicts
        next_agents = []
        for a in raw_agents:
            if isinstance(a, str):
                next_agents.append(a)
            elif isinstance(a, dict):
                # handle both {"name": "..."} and {"agent": "..."}
                name = a.get("name") or a.get("agent", "")
                next_agents.append(name)
            elif isinstance(a, list):
                next_agents.extend(x for x in a if isinstance(x, str))
        
        return cls(
            reasoning=parsed.get("reasoning", ""),
            mode=parsed.get("mode", "respond"),
            next_agents=next_agents,
            response=parsed.get("response"),
        )

# Orchestrator

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a warm, encouraging academic advisor for Rutgers CS students.
You coordinate specialist agents to answer student questions.

Each turn you receive a context snapshot and must respond with a JSON decision.

## Modes
- "route"    — call one or more agents to gather needed data. Only call an agent after its dependencies are met.
- "clarify"  — the query is entirely unrelated to course advising. Briefly redirect the student.
- "respond"  — you have enough data. Write a complete, helpful advisor-style response.

## Routing Rules
- Any mention of a subject, topic, or course name -> route immediately, never clarify.
- course_info intent -> call data_lookup.
- course_recommendation with no stated interests and no data yet -> ask the student what they're interested in (one friendly question, mode="clarify").
- If results are already collected -> respond immediately, never re-route.
- Never call an agent that has already been called.
- next_agents must be a flat list of strings e.g. ["data_lookup"]. Never dicts or nested lists.
- Ignore missing_critical_info in parsed data when interests are already present. 
  That field is for the parser's own confidence tracking, not a routing signal.
- If the student has already answered a clarifying question or repeated their 
  request, never ask for clarification again. Route with whatever information is available.
- If the student asks about course availability and courses are already present in "Courses Already Resolved This Session" with their offered status, 
  respond directly from that data. Do NOT call any agents.
- When responding about availability, always mention the specific semester (e.g. "Spring 2026") not just "this semester" or "next semester".
- course_recommendation intent MUST call planning after data_fetch completes. Never respond directly after data_fetch for recommendations — always route to planning first.
- If a course has unknown or unverified availability (offered status is null/None), you MUST call data_lookup to verify — never assume not offered.
- course_recommendation intent MUST call data_fetch first, NEVER data_lookup. data_lookup is ONLY for course_info intent when a specific course name or code is given.
- Never call data_lookup to verify availability of recommendation results — that is not its purpose. If availability is unknown, respond with whatever data_fetch returned and note that availability is unverified.
## Response Rules
- Only reference courses and prerequisites explicitly present in collected data. Never infer.
- The "response" field must be plain conversational text. Never JSON, code blocks, or markdown fences.
- For course recommmendations or planning, PLEASE use data and constraint agent to give you informed choice about course offerings and availibity. 

## Output Format (JSON only, no markdown fences)
{
  "reasoning": "<1-2 sentences>",
  "mode": "route" | "clarify" | "respond",
  "next_agents": [],
  "response": null
}

IMPORTANT: Your response must be valid JSON. Do not include literal newlines inside string values. Use \n for line breaks within strings.

When mode is "route": populate next_agents, set response to null.
When mode is "clarify" or "respond": next_agents must be [], response must be a non-null string.

IMPORTANT:
For course recommendation responses, space out each course clearly using this format:

1) Course Name (Course Code)
   Brief description.
   Prerequisites: X, Y, Z.

2) Next Course (Course Code)
   Brief description.
   Prerequisites: A, B.

Leave a blank line between each course.

Last thing: If you have a course code included in a response but can't match course code back to a name, don't assume the name of the course code, just use the code itself. 
"""

class OrchestratorExecutor(Executor):
    """
    Hub orchestrator with a single LLM agent that routes, clarifies, and responds.

    Flow:
      handle_request — entry point; runs the routing loop
      handle_result  — collects spoke results; re-enters the routing loop
    """

    MAX_ITERATIONS = 6

    def __init__(self, chat_client: OpenAIChatClient, model_id: str):
        super().__init__(id="orchestrator")
        self.agent = chat_client.as_agent(
            instructions=ORCHESTRATOR_SYSTEM_PROMPT,
            name="Orchestrator",
        )
        self.thread = self.agent.get_new_thread()

    # Entry point

    @handler
    async def handle_request(self, message: OrchestratorRequest, ctx: WorkflowContext[AgentResult]) -> None:
        await self._maybe_reset_thread(message.conversation_state)

        # print(f"[Orchestrator] parsed_data: {message.parsed_data}")
        intent = message.parsed_data.get("intent")
        has_transcript = bool(message.conversation_state.transcript_data)
        # print(f"[Orchestrator] intent='{intent}', has_transcript={has_transcript}")

        if intent == "transcript_upload":
            # print("[Orchestrator] Hard rule -> transcript agent")
            await ctx.send_message(AgentResult(
                message.user_query, message.parsed_data,
                agent_name="transcript", data={},
                conversation_state=message.conversation_state,
            ))
            return

        # Re-resolve semester if user mentions one in follow-up
        from agents.data_agent import DataAgent as DA
        if any(k in message.user_query.lower() for k in ["next", "spring", "fall", "summer", "winter"]):
            new_semester = DA.resolve_semester(None, message.user_query)
            if new_semester != message.conversation_state.resolved_semester:
                message.conversation_state.resolved_semester = new_semester
                # Save titles before clearing, so referential resolution still works
                saved_courses = {
                    code: v for code, v in message.conversation_state.resolved_courses.items()
                }
                message.conversation_state.resolved_courses = {}
                # Re-add with new semester but mark as unverified for re-lookup
                for code, v in saved_courses.items():
                    message.conversation_state.resolved_courses[code] = {**v, "offered": None}

        # Resolve referential courses
        entities = message.parsed_data.get("entities", {})
        if (not entities.get("specific_courses") and
            message.conversation_state.resolved_courses and
            any(ref in message.user_query.lower() for ref in ["those", "both", "all three", "these", "them", "they"])):
            entities["specific_courses"] = [
                v["title"] for v in message.conversation_state.resolved_courses.values()
            ]
        
        if intent == "clarification" and message.conversation_state.resolved_courses:
            entities["specific_courses"] = [
                v["title"] for v in message.conversation_state.resolved_courses.values()
            ]
        
        routing_ctx = RoutingContext(
            user_query=message.user_query,
            parsed_data=message.parsed_data,
            has_transcript=has_transcript,
            resolved_courses=message.conversation_state.resolved_courses,
            resolved_semester=message.conversation_state.resolved_semester or {},
        )

        # print(f"[DEBUG] routing_ctx prompt:\n{routing_ctx.to_prompt()}")
        await self._routing_loop(routing_ctx, message, ctx, iteration=0)

    # Spoke result collector

    @handler
    async def handle_result(self, message: AgentResult, ctx: WorkflowContext[AgentResult]) -> None:
        if message.agent_name == "transcript":
            return

        # print(f"[Orchestrator] Result from '{message.agent_name}'")

        routing_ctx: RoutingContext = message.conversation_state.routing_ctx
        routing_ctx.accumulated_results[message.agent_name] = message.data
        routing_ctx.has_transcript = bool(message.conversation_state.transcript_data)

        if message.agent_name == "data_fetch":
            semester = message.data.get("semester")
            if semester:
                message.conversation_state.resolved_semester = semester
            for course in message.data.get("courses", []):
                actual = course.get("course", course)
                code = actual.get("code")
                title = actual.get("title", "")
                if code:
                    message.conversation_state.resolve_course(
                        code, title, True,
                        message.conversation_state.resolved_semester or {}
                    )
                    # print(f"[Orchestrator] Cached resolved course: {code} - {title}")

        elif message.agent_name == "data_lookup":
            for course_result in message.data.get("courses", []):
                actual = course_result.get("course", {})
                code = actual.get("code")
                title = actual.get("title", "")
                offered = course_result.get("offered")
                semester = course_result.get("semester", message.conversation_state.resolved_semester or {})
                if code:
                    message.conversation_state.resolve_course(code, title, offered, semester)
                    # print(f"[Orchestrator] Cached lookup course: {code} - {title} (offered: {offered})")

        iteration: int = message.conversation_state.routing_iteration
        await self._routing_loop(routing_ctx, message, ctx, iteration=iteration)

    # Core routing loop

    async def _routing_loop(
        self,
        routing_ctx: RoutingContext,
        message,
        ctx: WorkflowContext,
        iteration: int,
    ) -> None:
        if iteration >= self.MAX_ITERATIONS:
            # print("[Orchestrator] Max iterations reached — forcing respond mode")
            await self._force_respond(routing_ctx, ctx)
            return

        # print(f"[Orchestrator] Routing iteration {iteration + 1}")
        raw = await self.agent.run(routing_ctx.to_prompt(), thread=None)  # keep thread for context, but don't feed prompt back in — rely on conversation history for context instead

        # print(type(raw))
        # print(hasattr(raw, "usage_details"))
        # print(getattr(raw, "usage_details", None))

        if hasattr(raw, "usage_details") and raw.usage_details:
            message.conversation_state.add_usage(
                raw.usage_details.get("input_token_count", 0) or 0,
                raw.usage_details.get("output_token_count", 0) or 0,
            )

        raw_text = raw.content if hasattr(raw, "content") else str(raw)

        try:
            decision = RoutingDecision.from_llm_output(raw_text)
            # print(f"[DEBUG] Orchestrator decision: mode={decision.mode}, agents={decision.next_agents}, reasoning={decision.reasoning}")
        except (json.JSONDecodeError, KeyError) as e:
            # print(f"[Orchestrator] Bad routing output: {e} — forcing respond mode")
            await self._force_respond(routing_ctx, ctx)
            return

        # print(f"[Orchestrator] mode={decision.mode}, reasoning: {decision.reasoning}")

        # Clarify or respond, yield output and stop
        if decision.mode in ("clarify", "respond"):
            if not decision.response:
                # print("[Orchestrator] Empty response in terminal mode — forcing respond")
                await self._force_respond(routing_ctx, ctx)
                return
            
            final_prompt = (
                f"{routing_ctx.to_prompt()}\n\n"
                f"Write your response to the student now. "
                f"Mode is '{decision.mode}'. Suggested response:\n{decision.response}"
            )
            final_raw = await self.agent.run(final_prompt, thread=self.thread)
            
            if hasattr(final_raw, "usage_details") and final_raw.usage_details:
                message.conversation_state.add_usage(
                    final_raw.usage_details.get("input_token_count", 0) or 0,
                    final_raw.usage_details.get("output_token_count", 0) or 0,
                )
            
            final_text = final_raw.content if hasattr(final_raw, "content") else str(final_raw)
            
            # Strip JSON wrapper if LLM still returned one
            try:
                parsed = RoutingDecision.from_llm_output(final_text)
                if parsed.response:
                    final_text = parsed.response
            except (json.JSONDecodeError, KeyError):
                pass  # plain text response, use as-is

            await ctx.yield_output(final_text)
            return

        # Route — validate and dispatch
        # print(f"[Orchestrator] Raw next_agents from LLM: {decision.next_agents}")
        valid_agents = [a.strip() for a in decision.next_agents if a.strip() and a.strip() in AGENT_REGISTRY]
        # in _routing_loop, after getting valid_agents, add:
        already_called = set(routing_ctx.accumulated_results.keys())
        duplicate_agents = [a for a in valid_agents if a in already_called]
        if duplicate_agents:
            # print(f"[Orchestrator] LLM tried to re-call already completed agents: {duplicate_agents} 0 forcing respond")
            await self._force_respond(routing_ctx, ctx)
            return
        valid_agents = [a for a in valid_agents if a not in already_called]
        # print(f"[Orchestrator] Valid agents: {valid_agents}")

        NEVER_ROUTE = {"transcript"} # never route transcript, only do it at request of user. 
        valid_agents = [a for a in valid_agents if a not in NEVER_ROUTE]

        valid_agents = valid_agents[:1]  # enforce sequential, only one agent at a time. 

        if not valid_agents:
            # print("[Orchestrator] No valid agents in route decision - forcing respond")
            await self._force_respond(routing_ctx, ctx)
            return

        # Stash routing context onto conversation state so handle_result can retrieve it
        message.conversation_state.routing_ctx = routing_ctx         
        message.conversation_state.routing_iteration = iteration + 1  

        # print(f"[Orchestrator] Dispatching -> {valid_agents}")
        for agent_name in valid_agents:
            routing_ctx.agents_call_order.append(agent_name)

            # Pass only what each agent needs
            if agent_name == "constraint_full":
                spoke_data = dict(routing_ctx.accumulated_results.get("data_fetch", {}))
            elif agent_name == "planning":
                spoke_data = {
                    "courses": routing_ctx.accumulated_results.get("constraint_full", {}).get("courses")
                            or routing_ctx.accumulated_results.get("data_fetch", {}).get("courses", []),
                    "constraint_data": routing_ctx.accumulated_results.get("constraint_full", {}).get("constraint_data", {}),
                }
                if agent_name == "planning" and not spoke_data.get("courses"): # cut ahead early, empty courses guard
                    # print("[Orchestrator] No courses to rank — responding directly")
                    await self._force_respond(routing_ctx, ctx)
                    return
                # print(f"[Orchestrator] Planning input courses: {[c.get('code') for c in spoke_data['courses']]}")

            else:
                spoke_data = dict(routing_ctx.accumulated_results)

            await ctx.send_message(AgentResult(
                routing_ctx.user_query,
                routing_ctx.parsed_data,
                agent_name=agent_name,
                data=spoke_data,
                conversation_state=message.conversation_state,
            ))

    # Fallback response when something goes wrong -> dump all data context and formulate best response. 

    async def _force_respond(self, routing_ctx: RoutingContext, ctx: WorkflowContext[AgentResult]) -> None:
        prompt = (
            f"{routing_ctx.to_prompt()}\n\n"
            f"Full collected data:\n{json.dumps(routing_ctx.accumulated_results, indent=2)}\n\n"
            f"You must now respond directly to the student based on whatever data is available."
        )
        # thread=None — force respond is a fallback, keep it lean
        raw = await self.agent.run(prompt, thread=None)
        raw_text = raw.content if hasattr(raw, "content") else str(raw)

        try:
            decision = RoutingDecision.from_llm_output(raw_text)
            text = decision.response if decision.mode in ("clarify", "respond") and decision.response else raw_text
        except (json.JSONDecodeError, KeyError):
            text = raw_text

        await ctx.yield_output(text)

    async def _maybe_reset_thread(self, conversation_state):
        self.thread = self.agent.get_new_thread()
    
        history = conversation_state.conversation_history
        if history:
            # Grab last 2 full turns (up to 4 messages: user+assistant pairs)
            recent = history[-4:]
            lines = "\n".join(
                f"{m['role'].upper()}: {m['content'][:400]}"
                for m in recent
            )
            summary = (
                f"You are continuing a conversation with a Rutgers CS student.\n\n"
                f"## Recent conversation\n{lines}\n\n"
                f"Transcript on file: {bool(conversation_state.transcript_data)}. "
                f"Use this context when writing your response to the student."
            )
            raw = await self.agent.run(summary, thread=self.thread)
            if hasattr(raw, "usage_details") and raw.usage_details:
                conversation_state.add_usage(
                    raw.usage_details.get("input_token_count", 0) or 0,
                    raw.usage_details.get("output_token_count", 0) or 0,
                )
        # print("[Orchestrator] Thread reset with summary")
        
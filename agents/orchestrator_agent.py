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
        "description": (
            "Parses and stores the student's uploaded transcript. "
            "Must be called first if the intent is transcript_upload. "
            "Enables constraint agents to verify eligibility."
        ),
        "terminal": True,
    },
    "data_fetch": {
        "description": (
            "Fetches a list of courses matching filters in parsed_data (subject, level, credits, etc.). "
            "Call this first for recommendation requests. Returns a course list."
        ),
        "terminal": False,
    },
    "data_lookup": {
        "description": (
            "Looks up detailed info for a specific course by ID or name. "
            "Call for course_info intent. Returns structured course details."
        ),
        "terminal": False,
    },
    "data_prereq": {
        "description": (
            "Retrieves prerequisite requirements for a specific course. "
            "Call for prerequisite_check intent. Returns prerequisite course list."
        ),
        "terminal": False,
    },
    "constraint_full": {
        "description": (
            "Validates a fetched course list against the student's transcript "
            "(completed courses, credits, GPA). Requires data_fetch result AND transcript. "
            "Do NOT call without both. Returns filtered/annotated course list."
        ),
        "terminal": False,
    },
    "constraint_prereq": {
        "description": (
            "Checks whether the student meets prerequisites for a specific course. "
            "Requires data_prereq result AND transcript. "
            "Do NOT call without both. Returns eligibility verdict."
        ),
        "terminal": False,
    },
    "planning": {
        "description": (
            "Ranks and selects top course recommendations from a course list. "
            "MUST be called for all course_recommendation intents. "
            "Requires data_fetch result. Call after constraint_full if transcript is available. "
            "Returns ranked list with reasoning."
        ),
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
    conversation_history: list[dict]
    accumulated_results: dict[str, dict] = field(default_factory=dict)
    agents_call_order: list[str] = field(default_factory=list)  # helps log dynamic agent calls by orchestrator. 

    def _slim_results(self) -> dict:
        slim = {}
        for key, val in self.accumulated_results.items():
            if isinstance(val, dict) and "courses" in val:
                slim[key] = {**val, "courses": [
                    {k: v for k, v in c.items() if k in ("code", "title", "credits")}
                    for c in val["courses"][:10]
                ]}
            else:
                slim[key] = val
        return slim

    def to_prompt(self) -> str:
        return f"""\
    ## Student Query
    {self.user_query}

    ## Parsed Intent & Entities
    {json.dumps(self.parsed_data, indent=2)}

    ## Context
    - Transcript on file: {self.has_transcript}

    ## Agents Available
    {AGENT_REGISTRY_SUMMARY}

    ## Results Collected So Far
    {json.dumps(self._slim_results(), indent=2) if self.accumulated_results else "None yet."}

    ## Agents Already Called (do NOT call these again)
    {list(self.accumulated_results.keys()) if self.accumulated_results else "None"}

    ## Conversation History (last 6 turns)
    {json.dumps(self.conversation_history[-6:], indent=2)}
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
You are an academic advisor agent manager for Rutgers CS students.
You coordinate specialist agents and respond directly to students.
Be warm, clear, and encouraging.

Each turn you receive a context snapshot and must output a JSON decision.

Modes:
- "route"   — you need more data; list which agents to call next in next_agents.
              You may call multiple agents in parallel if they have no dependency on each other.
              Read each agent's description carefully — never call an agent before
              its required inputs have been collected.
- "clarify" — the query is completely off-topic or unrelated to course advising entirely;
              write a short redirect in "response". Do NOT clarify just because a subject
              or course name could be more specific — act on it directly.
- "respond" — you have enough data to give a complete, helpful answer;
              write the full advisor-style response in "response", explaining
              WHY each recommendation suits the student where relevant.

Rules:
- A student mentioning any subject, topic, or course name is sufficient to route immediately.
  Never ask for clarification in these cases.
- If the student asks for general information on a topic, treat it as course_info and call data_lookup.
- If you already have sufficient data in "Results Collected So Far", set mode="respond" immediately.
  Do NOT re-call agents or route unnecessarily.
- next_agents must contain only plain agent name strings e.g. ["data_lookup"], never dicts or nested lists.
- If next_agents would be empty or no agents are left to call, set mode="respond".
- When responding, only reference courses and completed prerequisites that are explicitly present in the collected data. 
  Never infer or assume the student has completed a course unless it appears in their transcript data. If unsure, omit the claim.

Output format (JSON only, no markdown fences):
{
  "reasoning": "<1-2 sentences explaining your choice>",
  "mode": "route" | "clarify" | "respond",
  "next_agents": [],
  "response": null
}

When mode is "route", populate next_agents and leave response null.
When mode is "clarify" or "respond", next_agents must be [] and response must be a non-null string.
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

    # Entry point

    @handler
    async def handle_request(self, message: OrchestratorRequest, ctx: WorkflowContext) -> None:
        print(f"[Orchestrator] parsed_data: {message.parsed_data}")
        intent = message.parsed_data.get("intent")
        has_transcript = bool(message.conversation_state.transcript_data)
        print(f"[Orchestrator] intent='{intent}', has_transcript={has_transcript}")

        # Constraint rule: transcript upload bypasses LLM routing entirely
        if intent == "transcript_upload":
            print("[Orchestrator] Hard rule → transcript agent")
            await ctx.send_message(AgentResult(
                message.user_query, message.parsed_data,
                agent_name="transcript", data={},
                conversation_state=message.conversation_state,
            ))
            return

        routing_ctx = RoutingContext(
            user_query=message.user_query,
            parsed_data=message.parsed_data,
            has_transcript=has_transcript,
            conversation_history=message.conversation_state.conversation_history or [],
        )

        await self._routing_loop(routing_ctx, message, ctx, iteration=0)

    # Spoke result collector

    @handler
    async def handle_result(self, message: AgentResult, ctx: WorkflowContext) -> None:
        # Transcript spoke handles its own output — nothing to do
        if message.agent_name == "transcript":
            return

        print(f"[Orchestrator] Result from '{message.agent_name}'")

        routing_ctx: RoutingContext = message.conversation_state.routing_ctx  # type: ignore[attr-defined]
        routing_ctx.accumulated_results[message.agent_name] = message.data
        routing_ctx.has_transcript = bool(message.conversation_state.transcript_data)

        iteration: int = message.conversation_state.routing_iteration  # type: ignore[attr-defined]
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
            print("[Orchestrator] Max iterations reached — forcing respond mode")
            await self._force_respond(routing_ctx, ctx)
            return

        print(f"[Orchestrator] Routing iteration {iteration + 1}")
        raw = await self.agent.run(routing_ctx.to_prompt())
        raw_text = raw.content if hasattr(raw, "content") else str(raw)

        try:
            decision = RoutingDecision.from_llm_output(raw_text)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Orchestrator] Bad routing output: {e} — forcing respond mode")
            await self._force_respond(routing_ctx, ctx)
            return

        print(f"[Orchestrator] mode={decision.mode}, reasoning: {decision.reasoning}")

        # Clarify or respond — yield output and stop
        if decision.mode in ("clarify", "respond"):
            if not decision.response:
                print("[Orchestrator] Empty response in terminal mode — forcing respond")
                await self._force_respond(routing_ctx, ctx)
                return
            await ctx.yield_output(decision.response)
            return

        # Route — validate and dispatch
        print(f"[Orchestrator] Raw next_agents from LLM: {decision.next_agents}")
        valid_agents = [a.strip() for a in decision.next_agents if a.strip() and a.strip() in AGENT_REGISTRY]
        # in _routing_loop, after getting valid_agents, add:
        already_called = set(routing_ctx.accumulated_results.keys())
        duplicate_agents = [a for a in valid_agents if a in already_called]
        if duplicate_agents:
            print(f"[Orchestrator] LLM tried to re-call already completed agents: {duplicate_agents} — forcing respond")
            await self._force_respond(routing_ctx, ctx)
            return
        valid_agents = [a for a in valid_agents if a not in already_called]
        print(f"[Orchestrator] Valid agents: {valid_agents}")

        NEVER_ROUTE = {"transcript"} # never route transcript, only do it at request of user. 
        valid_agents = [a for a in valid_agents if a not in NEVER_ROUTE]

        valid_agents = valid_agents[:1]  # enforce sequential — one agent at a time

        if not valid_agents:
            print("[Orchestrator] No valid agents in route decision — forcing respond")
            await self._force_respond(routing_ctx, ctx)
            return

        # Stash routing context onto conversation state so handle_result can retrieve it
        message.conversation_state.routing_ctx = routing_ctx         
        message.conversation_state.routing_iteration = iteration + 1  

        print(f"[Orchestrator] Dispatching → {valid_agents}")
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

    async def _force_respond(self, routing_ctx: RoutingContext, ctx: WorkflowContext) -> None:
        prompt = (
            f"{routing_ctx.to_prompt()}\n\n"
            f"Full collected data:\n{json.dumps(routing_ctx.accumulated_results, indent=2)}\n\n"
            f"You must now respond directly to the student based on whatever data is available."
        )
        raw = await self.agent.run(prompt)
        raw_text = raw.content if hasattr(raw, "content") else str(raw)

        try:
            decision = RoutingDecision.from_llm_output(raw_text)
            if decision.mode in ("clarify", "respond") and decision.response:
                text = decision.response
            else:
                text = raw_text
        except (json.JSONDecodeError, KeyError):
            text = raw_text

        await ctx.yield_output(text)
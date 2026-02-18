"""
Orchestrator agent code
"""

import os
import json
from enum import Enum
from agent_framework.openai import OpenAIChatClient
from agent_framework import WorkflowEvent, WorkflowBuilder, WorkflowOutputEvent
from dotenv import load_dotenv
from agent_framework import ChatAgent

from .shared_types import AgentResponse, ConversationState
from .parser_agent import ParserAgent # Done, needs to be adjusted
from .data_agent import DataAgent # Done, needs to be adjusted
from .planning_agent import PlanningAgent # Done, needs to bne adjusted

try:
    from .constraint_agent import ConstraintAgent # Not done
    ALL_AGENTS_AVAILABLE=True
except ImportError as e:
    print(f"Not all agents imported yet. ")
    ALL_AGENTS_AVAILABLE = False
    ConstraintAgent = None

load_dotenv()

class OrchestratorAgent():

    ORCHESTRATOR_NAME = "Orchestrator"
    # ORCHESTRATOR_INSTRUCTIONS = """
    #     You are an experienced advisor who has much experience in assisting Computer Science students at Rutgers University, providing the best support when it comes to 
    #     helping the students look into potential courses they may want to take. Your job is to utilize a few other different agents in order to ensure you can return the best
    #     recommended courses for a student. The agents you will be using are:
    #     - Parser Agent: Validates queries and extracts intent/entities
    #     - Data Agent: Retrieves course data and student records
    #     - Constraint Agent: Validates prerequisites and requirements
    #     - Planning Agent: Generates ranked course recommendations

    #     In general, Your role is to:
    #     1. Coordinate multiple specialized agents to help students find the best courses
    #     2. Decide when to invoke each agent and in what order
    #     3. Handle the overall conversation flow with students
    #     4. Ensure recommendations meet all requirements
    #     5. Format final responses in a clear, helpful manner

    #     Prioritize precision, helpfulness, and success of the student. 
    #     """
    ORCHESTRATOR_INSTRUCTIONS = """ 
        You are an experienced advisor who has much experience in assisting Computer Science students at Rutgers University.

        You currently work with:
        - Parser Agent: Validates queries and extracts intent/entities
        - Data Agent: Retrieves relevant courses from the course catalog
        - Planning Agent: Ranks courses based on student preferences and needs

        Your role is to:
        1. Use the Parser Agent to understand student queries
        2. Use the Data Agent to find courses matching their interests
        3. Use the Planning Agent to rank the top 5 courses for the student
        4. Present the ranked recommendations in a clear, conversational way
        5. Explain WHY each course was ranked where it was based on student profile
        6. Handle the conversation flow with students

        NOTE: The Constraint Agent (prerequisites validation) is not yet available.
        You provide ranked recommendations but cannot yet verify prerequisites.

        Be helpful, clear, and encouraging. Prioritize student success.
        """
    
    # ABOVE TEMPORARY INSTRUCTIONS B.C. FULL NOT AVAILABLE YET. 

    
    
    def __init__(
        self, 
        base_url: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None
    ):
        
        self.base_url = base_url or os.environ.get("GITHUB_ENDPOINT")
        self.api_key = api_key or os.environ.get("GITHUB_TOKEN")
        self.model_id = model_id or os.environ.get("GITHUB_MODEL_ID")
        
        self.chat_client = OpenAIChatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model_id=self.model_id
        )

        self.agent = None
        self.parser_agent = None
    
    def create_orchestrator(self):
        """
        Creates orchestrator agent using ChatAgent
        """
        self.agent = ChatAgent(
            chat_client=self.chat_client,
            model=self.model_id,
            instructions = self.ORCHESTRATOR_INSTRUCTIONS
        )
        return self.agent

    def initialize_agents(self):
        self.create_orchestrator()

        try:
            print("Initialize parser agent...")
            self.parser_agent = ParserAgent(
                client=self.chat_client,
                model="gpt-4o-mini",
                schema_path="agents/query_schema.json"
            )
        except Exception as e:
            print(f"Failed to initialize ParserAgent: {e}")

        try:
            print("Initializing data agent...") # agent tools may be expanded
            self.data_agent = DataAgent(
                client=self.chat_client,
                model=self.model_id,
                courses_file="rutgers_courses.json" 
            )
        except Exception as e:
            print(f"Failed to initialize DataAgent: {e}")

        try:
            print("Initializing planning agent...")
            self.planning_agent = PlanningAgent(
                client=self.chat_client,
                model = self.model_id
            )
        except Exception as e:
            print(f"Failed to initialize PlanningAgent: {e}")

    def _format_history(self, state: ConversationState) -> str:
        """Helper to format conversation history for LLM input"""
        if not state.conversation_history:
            return ""
        
        history_str = "Previous conversation:\n"
        for msg in state.conversation_history:
            role = "Student" if msg['role'] == 'user' else "Advisor"
            history_str += f"{role}: {msg['content']}\n"
        return history_str + "\n"

    async def process_query(self, user_query: str, state: ConversationState) -> tuple[str, list]:
        """
        Processes user query
        """
        agents_invoked = []

        if self.parser_agent is not None:
            try:
                state.user_query = user_query
                print("\n[Orchestrator] Invoking Parser Agent...")
                agents_invoked.append("ParserAgent")

                parser_response = await self.parser_agent.parse(user_query, state)
                print(f"[Parser Agent] success: {parser_response.success}")

                if not parser_response.success:
                    return f"I had trouble understanding your request: {', '.join(parser_response.errors)}", agents_invoked
                
                parsed_data = parser_response.data

                if not parsed_data.get('is_course_related', False):
                    return "I'm here to help with course recommendations at Rutgers CS. Your question seems to be about something else. Can you ask me about courses, prerequisites, or class planning?", agents_invoked
                
                if parsed_data.get('intent') == 'off_topic':
                    return "I specialize in helping Rutgers CS students find courses. Could you ask me something about course selection?", agents_invoked
                    
                if parsed_data.get('needs_clarification', False):
                    suggestions = parsed_data.get('suggested_clarifications', [])
                    if suggestions:
                        # Uses LLM reasoning to learn missing info LLM needs and asks user for it. 
                        print(f"[Orchestrator] Optional clarifications noted: {suggestions}")

                        entities = parsed_data.get('entities', {})
                        interests = entities.get('interests', [])
        
                        # Force clarification for overly generic queries
                        if not interests or interests == ['Computer Science'] or interests == ['CS']: # kinda hardcoded for now. 
                            return f"I'd love to help you find courses! To give you the best recommendations, could you tell me:\n" + "\n".join(f"- {s}" for s in suggestions), agents_invoked
                
                print(f"[Parser Agent] Intent: {parsed_data.get('intent')}")
                print(f"[Parser Agent] Entities: {parsed_data.get('entities')}")

                if self.data_agent is None:
                        return "the data retrieval system is not available yet. ", agents_invoked
                   
                agents_invoked.append("Data Agent")
                data_response = await self.data_agent.fetch_courses(
                    parsed_data=parsed_data,
                    state=state
                )

                if not data_response.success:
                    return f"I couldn't retrieve course data: {', '.join(data_response.errors)}", agents_invoked
                
                courses = data_response.data.get('courses', [])
                print(f"[Data Agent] Found {len(courses)} relevant courses")

                # course ranking logic in orchestrator 

                if len(courses) == 0:
                    return "I couldn't find any courses matching your criteria. Could you try rephrasing your interests or being more specific?", agents_invoked
                
                print("\n[Orchestrator] Invoking Planning Agent...")
                agents_invoked.append("Planning Agent")
                planning_response = await self.planning_agent.rank_courses(
                    courses=courses,
                    parsed_data=parsed_data,
                    state=state,
                    max_results=5
                )

                if not planning_response.success:
                    return f"I had trouble ranking the courses: {', '.join(planning_response.errors)}", agents_invoked
                
                ranked_data = planning_response.data
                ranked_courses = ranked_data.get('ranked_courses', [])
                ranking_summary = ranked_data.get('ranking_summary', '')

                print(f"[PlanningAgent] Ranked {len(ranked_courses)} courses")
                print("\n[Orchestrator] Generating final response...")

                context = f"""
                {self._format_history(state)}
                Student Query: {user_query}

                Student Profile:
                - Year: {parsed_data.get('entities', {}).get('year')}
                - Interests: {parsed_data.get('entities', {}).get('interests')}
                - Career Path: {parsed_data.get('entities', {}).get('career_path')}
                - Difficulty Preference: {parsed_data.get('entities', {}).get('difficulty_preference')}
                - GPA Priority: {parsed_data.get('entities', {}).get('gpa_priority')}

                Ranking Summary: {ranking_summary}

                TOP 5 RECOMMENDED COURSES:
                {json.dumps(ranked_courses, indent=2)}

                Based on this information, provide a warm, conversational response to the student.

                Present the ranked recommendations clearly. For each course:
                1. State the rank and course (e.g., "#1 - CS 434: Machine Learning")
                2. Explain the reasoning in your own words (don't just copy the reasoning verbatim)
                3. Highlight key benefits for THIS student

                Use a friendly, advisor-like tone. Make it feel personal and encouraging.
                Remember: Prerequisites are not yet validated, so mention this if relevant.

                Keep your response conversational and helpful.
                """

                response = await self.agent.run(context)
                final_response = response.messages[-1].contents[0].text
                return final_response, agents_invoked
                    
            except Exception as e:
                print(f"[Orchestrator] Error in pipeline: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to basic response
                print("\n[Orchestrator] Falling back to basic response...")
                fallback_context = f"{self._format_history(state)}Student Query: {user_query}"
                response = await self.agent.run(fallback_context)
                return response.messages[-1].contents[0].text, agents_invoked
        
"""
Notes on code:
- Now includes Planning Agent for course ranking
- Pipeline: Orchestrator -> Parser → Data → Planning → Orchestrator Final Response
- Still no conversation memory (future improvement) -> slightly improved with conversation history formatting, but not true memory yet
- Still no prerequisite validation (waiting for ConstraintAgent)
- No safeguards against agent hallucination yet
- Not capable of holding a full back and forth conversation right now
- Ranking is flawed 
- Need to adjust code so that it only redirects user to ask CS course related questions. 
"""

# initialize agents expands later. 

# general note for this week, follow tutorial very closely. 

# may want to utilize mem0 (maybe azure ai search functions?)
#  consider scratchpad plugin.
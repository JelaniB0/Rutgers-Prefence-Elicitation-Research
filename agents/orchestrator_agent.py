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
from typing import Dict, List, Any


from .shared_types import AgentResponse, ConversationState
from .parser_agent import ParserAgent # Done, needs to be adjusted
from .data_agent import DataAgent # Done, needs to be adjusted
from .planning_agent import PlanningAgent # Done, needs to be adjusted
from .transcript_agent import TranscriptAgent # Done. 


try:
    from .constraint_agent import ConstraintAgent # Done
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

        try:
            print("Initializing transcript agent...")
            self.transcript_agent = TranscriptAgent(
                client=self.chat_client,
                model=self.model_id
            )
        except Exception as e:
            print(f"Failed to initialize TranscriptAgent: {e}")
            self.transcript_agent = None

        try:
            print("Initializing constraint agent...")
            self.constraint_agent = ConstraintAgent(
                client=self.chat_client,
                model=self.model_id
            )
        except Exception as e:
            print(f"Failed to initialize ConstraintAgent: {e}")
            self.constraint_agent = None

    def _format_history(self, state: ConversationState) -> str:
        """Helper to format conversation history for LLM input"""
        if not state.conversation_history:
            return ""
        
        history_str = "Previous conversation:\n"
        for msg in state.conversation_history:
            role = "Student" if msg['role'] == 'user' else "Advisor"
            history_str += f"{role}: {msg['content']}\n"
        return history_str + "\n"
    
    async def _handle_course_info(self, parsed_data: Dict, state: ConversationState) -> str:
        """
        Handle course information lookup requests

        Args:
            parsed_data: Output from parser agent with intent/entities
            state: Current conversation state
        """
        entities = parsed_data.get('entities', {})
        specific_course = entities.get('specific_courses', [])

        if not specific_course:
            return "Could you specify which course you're interested in? For example, you can ask about 'CS 101' or 'Introduction to Computer Science'."
        
        course_query = specific_course[0]  # Assuming we take the first mentioned course for simplicity
        print(f"[Orchestrator] Handling course info request for: {course_query}")

        lookup_response = await self.data_agent.lookup_course(course_query, state)

        if not lookup_response.success:
            error_msg = lookup_response.errors[0] if lookup_response.errors else "Unknown error"

            if "not found" in error_msg.lower():
                return f"I couldn't find a course called '{course_query}' in the Rutgers CS catalog. Could you double-check the course code or name? You can also try asking 'what courses are available in [topic]' to browse related courses."
            else:
                return f"I had trouble looking up that course: {error_msg}"
            
        data = lookup_response.data
    
        # Handle multiple matches - need disambiguation
        if data.get('needs_disambiguation'):
            courses = data.get('courses', [])
            response = f"I found {len(courses)} courses matching '{course_query}':\n\n"
            for course in courses:
                response += f"• **{course.get('code')}** - {course.get('title')}\n"
            response += "\nWhich one would you like to know more about? Just tell me the course code."
            return response
        
        # Single course found - format detailed response
        course = data.get('course')
        
        # Use the orchestrator LLM to format a nice response
        context = f"""
        {self._format_history(state)}
        
        Student Query: {state.user_query}
        
        COURSE INFORMATION:
        {json.dumps(course, indent=2)}
        
        Provide a helpful, conversational response about this course.
        
        Structure your response with:
        1. Course code and title (bold the course code)
        2. Brief, engaging description of what the course covers
        3. Prerequisites (if any) - mention them clearly
        4. Credit hours
        5. Key topics/skills covered (2-4 bullet points)
        6. Who this course is good for (e.g., "Great if you're interested in X" or "Recommended for Y students")
        
        Be friendly, informative, and conversational. Don't just list information - make it engaging.
        If the course seems relevant to common career paths or builds important skills, mention that.
        
        End with an offer to help: "Would you like recommendations for related courses?" or 
        "Let me know if you'd like to know more about prerequisites or related topics!"
        """
        
        response = await self.agent.run(context)
        return response.messages[-1].contents[0].text
    
    async def _handle_prereq_check(self, parsed_data: Dict, state: ConversationState) -> str:
        """
        Handle prerequisite check requests.

        Looks up the course(s) mentioned, runs constraint validation,
        and returns a direct conversational answer about what the student
        needs before they can enroll.

        Args:
            parsed_data: Output from ParserAgent with intent/entities
            state: Current conversation state

        Returns:
            Formatted string response to present to the student
        """
        entities = parsed_data.get("entities", {})
        
        target_course = entities.get("target_course")
        related_courses = entities.get("related_courses", [])

        # If no target course, fallback
        if not target_course:
            specific_courses = entities.get("specific_courses", [])
            if specific_courses:
                target_course = specific_courses[0]
                related_courses = specific_courses[1:]

        # Look up target course only
        print(f"[Orchestrator] Checking prerequisites for: {target_course}")
        lookup_response = await self.data_agent.lookup_course(target_course, state)

        if not lookup_response.success:
            return (
                f"I couldn't find a course called '{target_course}' in the Rutgers CS catalog. "
                f"Could you double-check the course code or name?"
            )

        data = lookup_response.data
        if data.get("needs_disambiguation"):
            courses = data.get("courses", [])
            msg = f"I found {len(courses)} courses matching '{target_course}':\n\n"
            for c in courses:
                msg += f"• **{c.get('code')}** - {c.get('title')}\n"
            msg += "\nWhich one did you mean? Just give me the course code."
            return msg

        course = data.get("course")

        # Run constraints only for target course
        constraint_data = None
        if self.constraint_agent:
            constraint_response = await self.constraint_agent.check_single_course(course=course, state=state)
            if constraint_response.success:
                constraint_data = constraint_response.data

        # Build transcript context if available
        transcript_context = ""
        if state.transcript_data:
            transcript_context = self.transcript_agent.summarize_for_prompt(state.transcript_data)

        # Build constraint context
        constraint_context = ""
        if constraint_data:
            constraint_context = f"""
            PREREQUISITE CHECK RESULT:
            - Student is eligible: {constraint_data.get('eligible')}
            - Met prerequisites: {constraint_data.get('met_prerequisites', [])}
            - Unmet prerequisites: {constraint_data.get('unmet_prerequisites', [])}
            - Reasoning: {constraint_data.get('reasoning')}
            - Pathway suggestion: {constraint_data.get('pathway_suggestion')}
            - Credit standing appropriate: {constraint_data.get('standing_eligible')}
            - Standing note: {constraint_data.get('standing_note', '')}
            """

        related_note = ""

        if related_courses:
            related_note += f"\nAlso, the user mentioned {', '.join(related_courses)}. Include notes about how these relate to {target_course} if relevant."
        else:
            related_note = ""

        context = f"""
        {self._format_history(state)}
        
        Student Query: {state.user_query}
        
        COURSE INFORMATION:
        {json.dumps(course, indent=2)}
        
        {transcript_context}
        {constraint_context}
        {related_note}
        
        The student is asking about prerequisites for this course.
        Provide a clear, conversational answer that:
        1. States the course name and code
        2. Lists what prerequisites are required
        3. Checks student transcript if available and explains eligibility
        4. Suggests pathway if prerequisites not met
        5. Mentions credit standing if relevant
        """

        response = await self.agent.run(context)
        return response.messages[-1].contents[0].text
            
        #  # If we checked multiple courses, join the responses
        # return "\n\n---\n\n".join(responses)
    
    async def load_transcript(self, pdf_path: str, state: ConversationState) -> str:
        """
        Call this before process_query() when a student uploads a transcript.
        Parses it and stores data in state so all agents can access it.
        """
        if self.transcript_agent is None:
            return "Transcript reading is not available right now."

        print(f"[Orchestrator] Loading transcript from: {pdf_path}")
        response = await self.transcript_agent.parse_transcript(pdf_path, state)

        if not response.success:
            return f"I couldn't read that transcript: {', '.join(response.errors)}"

        data = response.data

        completed_cs = [
            c for c in data.get("completed_courses", []) if ":198:" in c.get("code", "")
        ]
        in_progress_cs = [
            c for c in data.get("in_progress_courses", []) if ":198:" in c.get("code", "")
        ]

        completed_str = "\n".join(
            f"  - {c['code']}: {c['title']} ({c.get('grade', 'P')})"
            for c in completed_cs
        ) or "  - None found"

        in_progress_str = "\n".join(
            f"  - {c['code']}: {c['title']}"
            for c in in_progress_cs
        ) or "  - None found"

        return (
            f"Got it! I've read your transcript. Here's what I found:\n\n"
            f"**Year:** {data.get('year_standing')}\n"
            f"**GPA:** {data.get('cumulative_gpa')}\n"
            f"**Credits Completed:** {data.get('total_degree_credits')}\n\n"
            f"**CS Courses Completed:**\n{completed_str}\n\n"
            f"**CS Courses In Progress:**\n{in_progress_str}\n\n"
            f"I'll factor all of this in when making recommendations."
        )

    async def process_query(self, user_query: str, state: ConversationState) -> tuple[str, list]:
        """
        Processes user query
        """
        agents_invoked = []

        if getattr(state, 'awaiting_transcript_path', False):
            state.awaiting_transcript_path = False
            if not os.path.exists(user_query.strip()):
                return f"I couldn't find a file at '{user_query.strip()}'. Could you double-check the path?", agents_invoked
            agents_invoked.append("TranscriptAgent") 
            response = await self.load_transcript(user_query.strip(), state)
            return response, agents_invoked

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
                
                if parsed_data.get('intent') == 'course_info':
                    print("[Orchestrator] Routing to course information handler")
                    agents_invoked.append("DataAgent")
                    response = await self._handle_course_info(parsed_data, state)
                    return response, agents_invoked
                
                if parsed_data.get('intent') == 'transcript_upload':
                    file_path = parsed_data.get('entities', {}).get('file_path')
                    
                    if file_path:
                        # user gave us the path directly in their message
                        if not os.path.exists(file_path):
                            return f"I couldn't find a file at '{file_path}'. Could you double-check the path?", agents_invoked
                        
                        agents_invoked.append("TranscriptAgent")
                        response = await self.load_transcript(file_path, state)
                        return response, agents_invoked
                
                    else:
                        # user asked naturally but didn't provide a path yet
                        state.awaiting_transcript_path = True
                        return "Sure! Go ahead and drop the path to your transcript PDF.", agents_invoked
                
                if parsed_data.get('intent') == 'prerequisite_check':
                    print("[Orchestrator] Routing to prerequisite check handler")
                    agents_invoked.append("DataAgent")
                    if self.constraint_agent is not None:
                        agents_invoked.append("ConstraintAgent")
                    response = await self._handle_prereq_check(parsed_data, state)
                    return response, agents_invoked
                    
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
                   
                agents_invoked.append("DataAgent")
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
                agents_invoked.append("PlanningAgent")
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

                # Newly implemented constraint check via constraint agent. 
                constraint_context = ""
                if self.constraint_agent is not None:
                    agents_invoked.append("ConstraintAgent")
                    constraint_response = await self.constraint_agent.validate_courses(
                        courses=ranked_courses,
                        state=state
                    )
                    if constraint_response.success:
                        constraint_context = self.constraint_agent.summarize_for_prompt(
                            constraint_response.data
                        )
                        # Replace ranked_courses with only eligible ones, preserving order
                        eligible_codes = {
                            c.get("code") for c in constraint_response.data.get("eligible_courses", [])
                        }
                        ranked_courses = [c for c in ranked_courses if c.get("course_code") in eligible_codes]

                print(f"[PlanningAgent] Ranked {len(ranked_courses)} courses")
                print("\n[Orchestrator] Generating final response...")

                transcript_context = ""
                if state.transcript_data:
                    transcript_context = self.transcript_agent.summarize_for_prompt(state.transcript_data)

                context = f"""
                {self._format_history(state)}
                Student Query: {user_query}

                Student Profile:
                - Year: {parsed_data.get('entities', {}).get('year')}
                - Interests: {parsed_data.get('entities', {}).get('interests')}
                - Career Path: {parsed_data.get('entities', {}).get('career_path')}
                - Difficulty Preference: {parsed_data.get('entities', {}).get('difficulty_preference')}
                - GPA Priority: {parsed_data.get('entities', {}).get('gpa_priority')}

                {transcript_context}


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
                If transcript data is available, acknowledge what the student has already completed
                and avoid recommending courses they've taken or are currently enrolled in.

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
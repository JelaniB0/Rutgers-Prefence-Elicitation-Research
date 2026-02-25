# this agent class parses query responses and matches them to relevant aspects for planning agent, figuring out intent of query as well. 
"""
parser_agent.py
Parser agent that validates queries and extracts intent/entities
Uses a JSON schema to match queries against valid course recommendation patterns

RESPONSIBILITIES:
- Validate if queries are appropriate and course-related
- Detect intent (course_recommendation, clarification, general_question, etc.)
- Extract entities (year, interests, credit_hours, career_path, etc.)
- Match queries against JSON schema patterns
- Flag inappropriate content using Azure Content Safety (optional)

DOES NOT:
- Retrieve course data
- Make recommendations
- Validate constraints
"""

import os
import json
import re
from typing import Dict, Any
from dotenv import load_dotenv
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

# Import shared data classes, parser agent uses orchestrator defined response structure, can access conversation state.
from .shared_types import AgentResponse, ConversationState

class ParserAgent(ChatAgent):
    """
    Parser agent that validates queries and extracts structured information
    """
    
    def __init__(self, client: OpenAIChatClient, model: str, schema_path: str = None):
        """
        Initialize the parser agent
        
        Args:
            client: OpenAIChatClient instance
            model: Model ID to use -> currently uses GPT-4o-mini
            schema_path: Path to the JSON schema file
        """
        if schema_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(current_dir, "query_schema.json")

        super().__init__(
            chat_client=client,
            model=model,
            instructions=self._get_system_message()
        )
        
        self.model = model  # Store model for metadata
        self.schema_path = schema_path
        self.query_schema = self._load_schema()
        
        print(f"[ParserAgent] Initialized with model: {model}")
        print(f"[ParserAgent] Loaded query schema from: {schema_path}")
        print(f"[ParserAgent] Using LLM to match queries against schema")
    
    def _get_system_message(self) -> str:
        """Define the system message for the parser agent"""
        return """You are an intelligent query parser for a course recommendation system.

Your role is to understand student queries and extract structured information using your reasoning.

CORE PRINCIPLES:
- Be permissive: Assume course-related unless obviously not (weather, sports, etc.)
- Be helpful: Students asking for help, guidance, or saying they're confused ARE seeking course recommendations
- Be conservative with entities: Only extract what is clearly stated or strongly implied
- Distinguish between course_recommendation and course_info:
  * course_info: Student asks about a SPECIFIC course (e.g., "Tell me about CS 111", "What is Data Structures?")
  * course_recommendation: Student wants suggestions/recommendations (e.g., "What courses should I take?")

INTENT RECOGNITION:
- course_info: Requires specific_courses entity - student mentions a course code or name
- course_recommendation: Requires interests/year - student wants personalized suggestions
- off_topic: Clearly unrelated to CS courses

Always return valid JSON with your analysis.
"""
    
    def _load_schema(self) -> Dict:
        """
        Load the query schema from JSON file
        
        Returns:
            Dictionary containing the query schema
        """
        try:
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)
            print(f"[ParserAgent] Schema loaded successfully")
            return schema
        except FileNotFoundError:
            print(f"[ParserAgent] WARNING: Schema file not found at {self.schema_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"[ParserAgent] ERROR: Invalid JSON in schema file: {e}")
            return {}
    
    # Method used to parse queries. 
    async def parse(self, query: str, state: ConversationState) -> AgentResponse:
        """
        Main parsing method - validates and extracts information from query
        
        Args:
            query: User's input query
            state: Current conversation state
            
        Returns:
            AgentResponse with parsed data or error
        """
        print(f"[ParserAgent] Parsing query: '{query[:50]}...'")
        
        try:
           parsed_data = await self._llm_parse(query, state)
           print(f"[ParserAgent] Parsed - Intent: {parsed_data.get('intent')}, "
                  f"Confidence: {parsed_data.get('confidence'):.2f}")
           
           entities = parsed_data.get('entities', {})
           interests = entities.get('interests', [])

            # checks for if query is too vague or generic. 
           if interests and len(interests) == 1 and interests[0].lower() in ['computer science', 'cs']:
            parsed_data['needs_clarification'] = True
            parsed_data['confidence'] = min(parsed_data.get('confidence', 0.5), 0.65)
            if 'suggested_clarifications' not in parsed_data or not parsed_data['suggested_clarifications']:
                parsed_data['suggested_clarifications'] = [
                    "What specific CS topics interest you? (e.g., AI, cybersecurity, web development, databases)",
                    "What year are you in?",
                    "Are you exploring for a career path or general interest?"
                ]
       
            print(f"[ParserAgent] Parsed - Intent: {parsed_data.get('intent')}, "
                    f"Confidence: {parsed_data.get('confidence'):.2f}")
           
           return AgentResponse(
               success=True,
               data=parsed_data,
               metadata={
                   "model_used":self.model,
                   "parsing_method":"pure_llm"
               }
           )
        
        except Exception as e:
            print(f"[ParserAgent] Error during parsing: {str(e)}")
            return AgentResponse(
                success=False,
                data=None,
                errors=[f"Parsing error: {str(e)}"]
            )
           
    # quick validation method to filter out obviously invalid queries, won't be as robust or strong as LLM parse. 
    
    async def _llm_parse(self, query: str, state: ConversationState) -> Dict:
        """
        Use LLM  to analyze query against the schema
        Args - query, state
        Returns - Dictionary with compelete parsing analysis
        """

        context=""
        if state.conversation_history:
            recent_history = state.conversation_history[-2:]  # REDUCED from -4 to -2
            context = "\n\nRecent conversation context:\n"
            for msg in recent_history:
                context += f"{msg['role']}: {msg['content']}\n"
        
        # REMOVED the full schema dump - using condensed version instead
        schema_context = """
    VALID CS INTERESTS: AI, Machine Learning, Data Science, Web Development, Mobile Development, 
    Cybersecurity, Databases, Algorithms, Data Structures, Systems Programming, Operating Systems,
    Computer Networks, Theory, Software Engineering, Computer Graphics, Cloud Computing, DevOps, 
    Game Development, HCI, Robotics, NLP, Computer Vision, Compilers

    OFF-TOPIC SUBJECTS: political science, environmental science, biology, chemistry, physics, 
    history, economics, psychology, business, finance, weather, sports, entertainment

    COURSE CODE PATTERNS: "CS 111", "01:198:112", "Data Structures" (exact course names)
    """
        
        prompt = f"""Analyze this student query for a course recommendation system.

    Query: "{query}"{context}

    {schema_context}

    TASK: Analyze this query to determine intent and extract entities.

    INTENT CLASSIFICATION (choose one):
    - course_recommendation: User wants personalized course suggestions
    - course_info: User asks about a SPECIFIC course by code or name
    - prerequisite_check: User asks about prerequisites
    - clarification: User is providing additional info after being asked
    - general_question: General question about CS program
    - off_topic: Query is not CS-related

    CRITICAL RULES:
    1. course_info requires a specific course code or exact course name
    Examples: "Tell me about CS 111", "What's Data Structures about?"
    2. course_recommendation is for topic-based requests
    Examples: "What courses teach AI?", "I want to learn web development"
    3. If interests = only "Computer Science" or "CS" → needs_clarification=true
    4. Non-CS subjects → intent=off_topic, is_course_related=false
    5. Only extract CS-related interests from the valid list above

    ENTITY EXTRACTION:
    - year: freshman, sophomore, junior, senior, graduate (or null)
    - interests: List of CS topics from valid list (empty if none or only generic "CS")
    - specific_courses: Course codes or exact names (only if mentioned)
    - career_path: text description (or null)
    - gpa_priority: high, medium, low (or null)
    - difficulty_preference: easy, moderate, challenging (or null)
    - credit_hours: number (or null)
    - time_constraints: text description (or null)

    CONFIDENCE SCORING:
    - 0.8+: Clear intent and sufficient entities
    - 0.5-0.8: Clear intent but missing some entities
    - <0.5: Ambiguous or very incomplete

    Return ONLY this JSON structure (no extra text):

    {{
    "intent": "intent_name",
    "is_course_related": true/false,
    "confidence": 0.0-1.0,
    "needs_clarification": true/false,
    "reasoning": "brief explanation of your analysis",
    "entities": {{
        "year": null,
        "interests": [],
        "credit_hours": null,
        "career_path": null,
        "gpa_priority": null,
        "specific_courses": [],
        "prerequisites_taken": [],
        "difficulty_preference": null,
        "time_constraints": null
    }},
    "missing_critical_info": ["list of missing info"],
    "suggested_clarifications": ["specific questions to ask user"]
    }}
"""

        try:
            response = await self.run(prompt)
            last_message = response.messages[-1]
            response_text = last_message.contents[0].text

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                required_fields = ['intent', 'is_course_related', 'confidence', 'needs_clarification', 'entities']

                if all(field in parsed for field in required_fields):
                    return parsed
                else:
                    print(f"[ParserAgent] WARNING: No JSON found in LLM response")
                    print(f"[ParserAgent] Response: {response_text[:200]}")
                    return self._get_fallback_parse(query)
                
        except json.JSONDecodeError as e:
            print(f"[ParserAgent] JSON parsing error: {e}")
            return self._get_fallback_parse(query)
        except Exception as e:
            print(f"[ParserAgent] LLM parsing error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_parse(query)
    
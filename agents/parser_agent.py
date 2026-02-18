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
- Be confident: Provide confidence scores based on query clarity and completeness

You analyze queries holistically and make intelligent judgments about:
1. Whether the query is course-related
2. What the student is trying to accomplish (intent)
3. What information they've provided (entities)
4. What information is missing
5. How confident you are in your understanding

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
            recent_history = state.conversation_history[-4:]
            context = "\n\nRecent conversation context:\n"
            for msg in recent_history:
                context += f"{msg['role']}: {msg['content']}\n"
        
        schema_context=""
        if self.query_schema:
            schema_context = f"\n\nREFERENCE SCHEMA:\n{json.dumps(self.query_schema, indent=2)}\n"
            schema_context += "\nUse this schema to guide your analysis. Match the query against valid intents, check for invalid patterns, and extract entities according to the definitions."
        
        prompt = f"""Analyze this student query for a course recommendation system.

        Query: "{query}"{context}{schema_context}

        TASK: Analyze this query using the provided schema as your guide.

        INTENT CLASSIFICATION:
        Match the query against the valid_intents in the schema. Determine what the student is trying to accomplish.
        If the query matches invalid_query_patterns, mark it as off_topic.

        DOMAIN RESTRICTION:
        This system ONLY handles Rutgers Computer Science courses.
        Use the schema's "interests.valid_categories" as the definitive list of valid topics.
        Use the schema's "course_related_keywords" to determine if a query is course-related.

        If the student's interests do not map to anything in "interests.valid_categories", set:
        - is_course_related: false
        - intent: "off_topic"

        Examples of what should be off_topic:
        - Environmental Science → not in valid_categories → off_topic
        - Biology, Chemistry, History, Economics → off_topic
        - Machine Learning, Cybersecurity, Web Development → valid, proceed normally

        KEY RULES:
        - Students saying "I need help", "I don't know where to start", "What should I take?" are seeking course recommendations
        - Only mark as off_topic if clearly unrelated to courses/academics (weather, sports, entertainment, etc.)
        - Use the schema's intent_definitions to understand what each intent requires
        - IMPORTANT: If interests are only generic "Computer Science" or "CS" with no specific topics, set needs_clarification=true
        - IMPORTANT: Queries like "I like CS" or "I want CS courses" are TOO VAGUE and require clarification

        SPECIFICITY REQUIREMENTS:
        - Generic interest in "Computer Science" alone is NOT sufficient
        - Students must specify: AI, cybersecurity, web dev, databases, systems, theory, etc.
        - If confidence < 0.70 due to vagueness, set needs_clarification=true
        - If only interest extracted is "Computer Science" with nothing else, set needs_clarification=true

        ENTITY EXTRACTION:
        Extract entities according to the entity_definitions in the schema.
        - Only extract what is clearly stated or strongly implied
        - Follow the type and valid_values constraints in the schema
        - For categorical entities, only use values listed in valid_values
        - DO NOT extract "Computer Science" as an interest unless it's the only thing mentioned - in that case, flag for clarification

        CONFIDENCE ASSESSMENT:
        Calculate confidence (0.0 to 1.0) based on:
        - How well the query matches schema patterns
        - How many required entities for the intent are present
        - How clear and complete the query is
        - Use the confidence_calculation section of the schema as guidance

        SCHEMA MATCHING:
        - Identify which intent_definition this query best matches
        - Note which required_entities are present vs missing
        - Calculate a match_score based on completeness

        Return ONLY this JSON structure:

        {{
        "intent": "from valid_intents in schema or 'off_topic'",
        "is_course_related": true/false,
        "confidence": 0.0-1.0,
        "needs_clarification": true/false,
        "reasoning": "explain how you matched this query to the schema",
        
        "entities": {{
            "year": "value from schema valid_values or null",
            "interests": ["topic1", "topic2"] or null,
            "credit_hours": number or null,
            "career_path": "description" or null,
            "gpa_priority": "value from schema valid_values or null",
            "specific_courses": ["COURSE_CODE"] or null,
            "prerequisites_taken": ["COURSE_CODE"] or null,
            "difficulty_preference": "value from schema valid_values or null",
            "time_constraints": "description" or null
        }},
        
        "schema_match": {{
            "matched_intent_definition": "which intent definition from schema",
            "required_entities_present": ["entities that were required and found"],
            "required_entities_missing": ["entities that were required but not found"],
            "match_score": 0.0-1.0,
            "explanation": "why this match score"
        }},
        
        "missing_critical_info": ["What key information would help you better assist this student?"],
        "suggested_clarifications": ["Specific questions you could ask to get this info"]
        }}

        Use your intelligence to interpret the schema and make smart matching decisions.
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
    
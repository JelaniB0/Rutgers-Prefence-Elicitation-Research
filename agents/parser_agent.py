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
    
    # Match parsed data against schema patterns

    # LLM extracts query data, parses that data into structured formatting, then uses this hard-coded schema to match against query_schema.json with queries we are expecting or looking
    # for. Is this better? In terms of cost and speed? As well as LLM understanding not being a factor and us already knowing what we are looking for/expecting? 
    # Or is this worst because we are not leveraging how flexible LLM can be with different queries? 
    # Can consider a bit of both -- need to do further testing with this method and our confidence scoring method. 
    # def _match_schema(self, parsed_data: Dict) -> Dict:
    #     """
    #     Match parsed data against the query schema
    #     Uses NEW schema structure with intent_definitions
        
    #     Args:
    #         parsed_data: Parsed data from LLM
            
    #     Returns:
    #         Dictionary with schema matching results
    #     """
    #     intent = parsed_data.get("intent", "unknown")
    #     entities = parsed_data.get("entities", {})
        
    #     # Check if intent is valid
    #     valid_intents = self.query_schema.get("valid_intents", [])
    #     is_valid_intent = intent in valid_intents
        
    #     # Get required entities for this intent from NEW schema structure
    #     intent_definitions = self.query_schema.get("intent_definitions", {})
    #     intent_def = intent_definitions.get(intent, {})
    #     required_entities = intent_def.get("required_entities", [])
        
    #     # Check which entities are present
    #     present_entities = [key for key, value in entities.items() 
    #                       if value is not None and value != [] and value != ""]
        
    #     # Find missing required entities
    #     missing_required = [e for e in required_entities if e not in present_entities]
        
    #     # Calculate match score
    #     if is_valid_intent:
    #         if not missing_required:
    #             match_score = 1.0  # Perfect match
    #         elif required_entities:
    #             # Partial match based on how many required entities are present
    #             present_required = len(required_entities) - len(missing_required)
    #             match_score = 0.3 + (0.5 * (present_required / len(required_entities)))
    #         else:
    #             match_score = 0.8  # Valid intent but no required entities defined
    #     else:
    #         match_score = 0.0  # Invalid intent
        
    #     return {
    #         "is_valid_intent": is_valid_intent,
    #         "match_score": match_score,
    #         "missing_required_entities": missing_required,
    #         "present_entities": present_entities,
    #         "required_entities": required_entities
    #     }
    
    
    # # skeptical about this. 
    # def _calculate_confidence(self, parsed_data: Dict, schema_match: Dict) -> float:
    #     """
    #     Calculate overall confidence score for the parse
    #     Uses weights from schema if available
        
    #     Args:
    #         parsed_data: Parsed data from LLM
    #         schema_match: Schema matching results
            
    #     Returns:
    #         Confidence score between 0.0 and 1.0
    #     """
    #     # Get confidence calculation weights from schema
    #     confidence_config = self.query_schema.get("confidence_calculation", {})
    #     weights = confidence_config.get("weights", { # How do we correctly set weights here? 
    #         "intent_clarity": 0.3,
    #         "entity_completeness": 0.3,
    #         "course_keyword_presence": 0.2,
    #         "schema_match": 0.2
    #     })
        
    #     # 1. Intent clarity (schema match score)
    #     intent_clarity = schema_match["match_score"]
        
    #     # 2. Entity completeness
    #     entities = parsed_data.get("entities", {})
    #     non_null_entities = sum(1 for v in entities.values() 
    #                            if v is not None and v != [] and v != "")
    #     total_possible_entities = len(entities)
        
    #     if total_possible_entities > 0:
    #         entity_completeness = non_null_entities / total_possible_entities
    #     else:
    #         entity_completeness = 0.0
        
    #     # 3. Course keyword presence
    #     course_related = 1.0 if parsed_data.get("is_course_related", True) else 0.0
        
    #     # 4. Schema match (already calculated)
    #     schema_match_score = schema_match["match_score"]
        
    #     # Calculate weighted confidence
    #     confidence = (
    #         intent_clarity * weights.get("intent_clarity", 0.3) +
    #         entity_completeness * weights.get("entity_completeness", 0.3) +
    #         course_related * weights.get("course_keyword_presence", 0.2) +
    #         schema_match_score * weights.get("schema_match", 0.2)
    #     )
        
    #     # Boost if all required entities are present
    #     if not schema_match.get("missing_required_entities", []):
    #         confidence = min(1.0, confidence + 0.1)
        
    #     return round(confidence, 2)

# Code to consider LLM with hardcoded schema approach. 

# async def _llm_parse_with_schema(self, query: str, state: ConversationState) -> Dict:
#     """LLM sees full schema and makes intelligent decisions"""
    
#     prompt = f"""<query_schema>
# {json.dumps(self.query_schema, indent=2)}
# </query_schema>

# Parse this query according to the schema above.
# Query: "{query}"

# IMPORTANT:
# - Choose intent from valid_intents
# - Extract entities matching entity_definitions
# - Check against invalid_query_patterns
# - Explain your reasoning

# Return ONLY JSON."""
    
#     response = await self.run(prompt)
#     return self._extract_json(response)

# def _validate_llm_output(self, parsed: Dict) -> Dict:
#     """Python does safety validation on LLM output"""
    
#     # Safety: ensure intent is valid
#     if parsed["intent"] not in self.query_schema["valid_intents"]:
#         parsed["intent"] = "unknown"
#         parsed["validation_error"] = "Invalid intent from LLM"
    
#     # Safety: ensure entity types match
#     # (but don't be too strict - let LLM be smart)
    
#     return parsed
"""
Planning agent file

Currently uses LLM directly, will replace LLM logic with direct algorithmic logic later.
"""

import os
import json
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

from .shared_types import AgentResponse, ConversationState

class PlanningAgent(ChatAgent):
    """
    Planning agent to rank courses with LLM reasoning (placeholder for algorithm later)
    """

    def __init__(self, client: OpenAIChatClient, model: str):
        """
        Initialize planning agent

        Args - client: OpenAIChatClient instance, model: Model ID to use
        """

        super().__init__(
            chat_client=client,
            model=model,
            instructions=self._get_system_message()
        )

        self.model = model
        print(f"[PlanningAgent] Initialized with model: {model}, using LLM-based ranking")

    def _get_system_message(self) -> str:
        """System message for planning agent"""
        return """You are an expert course planning advisor for Rutgers Computer Science students.

        Your role is to rank courses based on student needs and preferences.

        RANKING CRITERIA (in order of importance):
        1. Relevance to student's interests and career goals
        2. Appropriate difficulty for student's year level
        3. Course quality and student reviews (when available)
        4. Skills development and learning outcomes
        5. Prerequisites completion (assume valid courses are provided)

        APPROACH:
        - Be thoughtful and analytical in your rankings
        - Consider the student holistically (year, interests, goals)
        - Provide clear reasoning for each ranking decision
        - Prioritize courses that best match the student's stated interests

        Always return valid JSON with your rankings and reasoning.
        """
    
    """
    Rank courses based on student preferences

    Args - courses (list of course dictionaries from data agent), parsed_data(parsed query data from parser agent), state(Current conversation state), max_results(max # of courses to rank)
    """
    async def rank_courses(self, courses: List[Dict], parsed_data: Dict, state: ConversationState, max_results: int = 5) -> AgentResponse:   
        try:
            num_to_rank = min(len(courses), max_results)
            if num_to_rank == 0:
                return AgentResponse(
                    success=False,
                    data=None,
                    errors=["No courses provided to rank"]
                )
            
            ranked_data = await self._llm_rank(courses, parsed_data, state, num_to_rank)

            print(f"[PlanningAgent] Successfully ranked {len(ranked_data.get('ranked_courses', []))} courses")

            return AgentResponse(
                success=True,
                data=ranked_data,
                metadata={
                    "model_used": self.model,
                    "ranking_method": "llm_baseline",
                    "total_courses_considered": len(courses),
                    "courses_ranked": num_to_rank
                }
            )
        
        except Exception as e:
            print(f"[PlanningAgent] ERROR during ranking: {str(e)}")
            import traceback
            traceback.print_exc()
            return AgentResponse(
                success=False,
                data=None,
                errors=[f"Ranking error: {str(e)}"]
            )
        
    async def _llm_rank(self, courses: List[Dict], parsed_data: Dict, state: ConversationState, max_results: int) -> Dict:
        entities = parsed_data.get('entities',{})
        student_year = entities.get('year', 'unknown')
        interests = entities.get('interests', [])
        career_path = entities.get('career_path')
        difficulty_pref = entities.get('difficulty_preference')
        gpa_priority = entities.get('gpa_priority')

        context = ""
        if state.conversation_history:
            recent_history = state.conversation_history[-2:]
            context = "\n\nRecent conversation:\n"
            for msg in recent_history:
                context+=f"{msg['role']}: {msg['content']}\n"
        
        courses_json = json.dumps(courses, indent=2)
        prompt = f"""Rank these Rutgers CS courses for a student based on their profile and preferences.

        STUDENT PROFILE:
        - Year: {student_year}
        - Interests: {', '.join(interests) if interests else 'Not specified'}
        - Career Path: {career_path or 'Not specified'}
        - Difficulty Preference: {difficulty_pref or 'Not specified'}
        - GPA Priority: {gpa_priority or 'Not specified'}
        {context}

        AVAILABLE COURSES:
        {courses_json}

        TASK: Select and rank the TOP {max_results} courses that best match this student's profile.

        RANKING CRITERIA:
        1. **Interest Match** - How well does the course align with stated interests?
        2. **Year Appropriateness** - Is this suitable for their academic level?
        3. **Career Relevance** - Does this support their career goals?
        4. **Learning Value** - What important skills/knowledge does this provide?
        5. **Student Preferences** - Does this match their difficulty/GPA preferences?

        For each ranked course, provide:
        - Clear reasoning for the ranking
        - Why it's a good match for THIS student specifically
        - What makes it better than courses ranked lower

        Return ONLY this JSON structure:

        {{
        "ranked_courses": [
            {{
            "rank": 1,
            "course_code": "CS XXX",
            "course_title": "Course Name",
            "reasoning": "Why this is the #1 recommendation for this student",
            "match_score": 0.0-1.0,
            "key_benefits": ["benefit 1", "benefit 2", "benefit 3"]
            }},
            ... (up to {max_results} courses)
        ],
        "ranking_summary": "Brief overall explanation of the ranking strategy used",
        "not_recommended": [
            {{
            "course_code": "CS XXX",
            "reason": "Why this course wasn't in top {max_results}"
            }}
        ]
        }}

        Be analytical and student-focused. Explain your reasoning clearly.
        """

        try:
            response = await self.run(prompt)
            last_message = response.messages[-1]
            response_text = last_message.contents[0].text

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                ranked = json.loads(json_match.group())

                if 'ranked_courses' in ranked and isinstance(ranked['ranked_courses'], list):
                    return ranked
                else:
                    print(f"[PlanningAgent] Warning: Invalid ranking structure")
                    return {}
            else:
                print(f"[PlanningAgent] WARNING: No JSON found in LLM response")
                print(f"[PlanningAgent] Response: {response_text[:200]}")
                return {}
            
        except json.JSONDecodeError as e:
            print(f"[PlanningAgent] JSON parsing error: {e}")
            return {}
        except Exception as e:
            print(f"[PlanningAgent] LLM ranking error: {e}")
            return {}



        

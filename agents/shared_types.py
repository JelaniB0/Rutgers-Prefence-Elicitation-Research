"""
shared_types.py
Shared data structures used across multiple agents to avoid circular imports

These classes are imported by all agents to ensure consistent data structures
without creating circular dependencies between agent modules.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ConversationState:
    """
    Tracks the state of an ongoing conversation with the user
    
    Attributes:
        user_query: The current user query being processed
        conversation_history: List of previous messages in the conversation
        extracted_entities: Entities extracted from the conversation so far
        current_intent: The detected intent of the current query
        user_profile: Information about the user (year, major, etc.)
        recommendations: Course recommendations made so far
        clarification_needed: Fields that need clarification from user
        session_id: Unique identifier for this conversation session
    """
    
    def __init__(
        self,
        user_query: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        transcript_data: Optional[Dict[str, Any]] = None
    ):
        self.user_query = user_query
        self.conversation_history = conversation_history or []
        self.session_id = session_id
        self.transcript_data = transcript_data
        self.last_intent = ""
        self.resolved_semester = None
        self.resolved_courses: Dict[str, Dict] = {}
        self.MAX_HISTORY = 12
        self.input_tokens = 0
        self.output_tokens = 0

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def reset_usage(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        # Keep only last N messages to avoid token overflow
        if len(self.conversation_history) > self.MAX_HISTORY:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY:]
    
    def update_entities(self, new_entities: Dict[str, Any]):
        """Update extracted entities with new values, only if they are not None/empty"""
        for key, value in new_entities.items():
            if value is not None and value != [] and value != "":
                self.extracted_entities[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "user_query": self.user_query,
            "conversation_history": self.conversation_history,
            "session_id": self.session_id,
            "transcript_data": self.transcript_data
        }
    
    def resolve_course(self, code: str, title: str, offered: bool, semester: str):
        """Cache a resolved course — minimal data only to save tokens"""
        self.resolved_courses[code] = {
            "title": title,
            "offered": offered,
            "semester": semester
        }

    def get_resolved_course(self, code: str) -> Optional[Dict]:
        return self.resolved_courses.get(code)

    def is_course_resolved(self, code: str) -> bool:
        return code in self.resolved_courses


@dataclass
class AgentResponse:
    """
    Standard response structure returned by all agents
    
    Attributes:
        success: Whether the agent operation succeeded
        data: The main data payload from the agent (parsed info, recommendations, etc.)
        errors: List of error messages if operation failed
        metadata: Additional metadata about the response (model used, timestamp, etc.)
        next_action: Suggested next action for the orchestrator
        requires_user_input: Whether user input is needed before proceeding
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    next_action: Optional[str] = None
    requires_user_input: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "metadata": self.metadata,
            "next_action": self.next_action,
            "requires_user_input": self.requires_user_input
        }
    
    def __str__(self) -> str:
        """String representation for logging"""
        if self.success:
            return f"AgentResponse(success=True, next_action={self.next_action})"
        else:
            return f"AgentResponse(success=False, errors={self.errors})"


class AgentType(Enum):
    """
    Enumeration of different agent types in the system
    """
    ORCHESTRATOR = "orchestrator"
    PARSER = "parser"
    DATA = "data"
    CONSTRAINT = "constraint"
    PLANNING = "planning"


class IntentType(Enum):
    """
    Enumeration of possible query intents
    """
    COURSE_RECOMMENDATION = "course_recommendation"
    CLARIFICATION = "clarification"
    GENERAL_QUESTION = "general_question"
    PREREQUISITE_CHECK = "prerequisite_check"
    SCHEDULE_PLANNING = "schedule_planning"
    OFF_TOPIC = "off_topic"
    TRANSCRIPT_UPLOAD = "transcript_upload"
    UNKNOWN = "unknown"


@dataclass
class CourseRecommendation:
    """
    Structure for a single course recommendation
    
    Attributes:
        course_code: The course identifier (e.g., "CS 101")
        course_name: Full name of the course
        credits: Number of credit hours
        reason: Why this course is recommended
        confidence: Confidence score for this recommendation (0.0 to 1.0)
        prerequisites: List of prerequisite courses
        difficulty: Estimated difficulty level
        relevance_score: How relevant to user's interests (0.0 to 1.0)
    """
    course_code: str
    course_name: str
    credits: int
    reason: str
    confidence: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    difficulty: str = "moderate"
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "course_code": self.course_code,
            "course_name": self.course_name,
            "credits": self.credits,
            "reason": self.reason,
            "confidence": self.confidence,
            "prerequisites": self.prerequisites,
            "difficulty": self.difficulty,
            "relevance_score": self.relevance_score
        }


@dataclass
class ConstraintViolation:
    """
    Structure for constraint violations
    
    Attributes:
        constraint_type: Type of constraint violated
        severity: How severe the violation is (high/medium/low)
        message: Description of the violation
        affected_courses: List of courses involved in the violation
        suggestion: Suggested fix for the violation
    """
    constraint_type: str
    severity: str
    message: str
    affected_courses: List[str] = field(default_factory=list)
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "constraint_type": self.constraint_type,
            "severity": self.severity,
            "message": self.message,
            "affected_courses": self.affected_courses,
            "suggestion": self.suggestion
        }
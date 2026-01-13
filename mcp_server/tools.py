from fastmcp import FastMCP
from typing import List
from mcp_server.schemas import StudentProfile, Course
from utils.llm_client import chat
import json

mcp = FastMCP("CS Planning MCP Server")

@mcp.resource("course_catalog://{school}")
def load_course_catalog(school: str) -> str:
    """Load course catalog for a given school."""
    courses = [
        {"course_id": "CS101", "name": "Intro to CS", "prerequisites": []},
        {"course_id": "CS102", "name": "Data Structures", "prerequisites": ["CS101"]},
        {"course_id": "CS103", "name": "Algorithms", "prerequisites": ["CS102"]},
    ]
    return json.dumps(courses)

@mcp.tool()
def analyze_prerequisites(completed: List[str]) -> dict:
    """Analyze which courses are missing based on completed courses."""
    required_courses = {"CS101", "CS102", "CS103"}
    missing = list(required_courses - set(completed))
    return {"missing_courses": missing}

@mcp.tool()
def chat_tool(messages: list[dict]) -> str:
    """
    Expose the dual-mode chat function to MCP clients.
    """
    return chat(messages)
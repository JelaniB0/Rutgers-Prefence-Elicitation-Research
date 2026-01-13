import requests
from mcp_server.schemas import StudentProfile
from utils.llm_client import chat

class MCPClientAgent:
    """
    AI Agent that uses LLM for reasoning and calls MCP server tools.
    """

    def __init__(self, server_url: str):
        self.base_url = server_url.rstrip("/")
        self.messages = []

        self.system_prompt = (
            "You are a planning agent for a CS course advising system.\n"
            "You do not recommend courses yet.\n"
            "Your responsibilities:\n"
            "- Extract structured information from the student\n"
            "- Identify missing or uncertain information\n"
            "- Analyze prerequisite satisfaction\n"
            "- Generate clarification questions\n"
            "- Prepare inputs for a future recommendation system\n"
            "Keep responses concise and focused."
        )

        self.messages.append({"role": "system", "content": self.system_prompt})

    def observe(self, user_input: str):
        """Record user input for LLM context."""
        self.messages.append({"role": "user", "content": user_input})

    def think(self):
        """Use LLM to reason about the situation."""
        assistant_msg = chat(self.messages)
        self.messages.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg

    def analyze(self, profile: StudentProfile):
        """Call MCP server tool to analyze prerequisites."""
        completed_ids = [c.course_id for c in profile.completed_courses]
        
        analyze_url = f"{self.base_url}/tools/analyze_prerequisites"
        response = requests.post(analyze_url, json={"completed": completed_ids})
        response.raise_for_status()
        return response.json()

    def get_catalog(self, school: str):
        """Get course catalog from MCP server."""
        catalog_url = f"{self.base_url}/resources/course_catalog/{school}"
        response = requests.get(catalog_url)
        response.raise_for_status()
        return response.json()
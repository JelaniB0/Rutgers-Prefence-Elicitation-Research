from mcp_client import MCPClientAgent
from mcp_server.schemas import StudentProfile, Course

# Connect to the HTTP server
agent = MCPClientAgent("http://127.0.0.1:5000")

# Example interaction
agent.observe("I have completed CS101 and CS102. What should I do next?")
response = agent.think()
print("LLM response:", response)

# Example student profile
profile = StudentProfile(
    school="CSSchool",
    completed_courses=[
        Course(course_id="CS101", name="Intro to CS"),
        Course(course_id="CS102", name="Data Structures")
    ]
)

# Analyze prerequisites
gaps = agent.analyze(profile)
print("\nPrerequisite gaps:", gaps)

# Get course catalog
catalog = agent.get_catalog("CSSchool")
print("\nCourse catalog:", catalog)
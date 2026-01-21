# pip install -r requirements.txt
import os
import asyncio
from dotenv import load_dotenv
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

# Load environment variables
load_dotenv()

url= os.environ.get("GITHUB_ENDPOINT", "https://models.inference.ai.azure.com"),
key= os.environ["GITHUB_TOKEN"],
model= os.environ["GITHUB_MODEL_ID"]


# Rutgers CS Major Core Courses Starter Data
COURSES = [
    {
        "code": "01:198:111",
        "name": "Introduction to Computer Science",
        "credits": 4,
        "prerequisites": [],
        "description": (
            "Intensive introduction to CS with problem decomposition, programming "
            "in Java, recursive algorithms, and basic data structures."
        ),
    },
    {
        "code": "01:198:112",
        "name": "Data Structures",
        "credits": 4,
        "prerequisites": ["01:198:111"],
        "description": (
            "Study of data abstraction and organization: lists, stacks, queues, "
            "trees, sorting/searching, and complexity analysis."
        ),
    },
    {
        "code": "01:198:205",
        "name": "Discrete Structures I",
        "credits": 3,
        "prerequisites": ["01:198:111"],
        "description": (
            "Introduction to discrete mathematical structures for computer science, "
            "including logic, sets, and proof techniques."
        ),
    },
    {
        "code": "01:198:206",
        "name": "Discrete Structures II",
        "credits": 3,
        "prerequisites": ["01:198:205"],
        "description": (
            "Continuation of discrete mathematics: counting, relations, graph theory, "
            "and recurrence relations."
        ),
    },
    {
        "code": "01:198:211",
        "name": "Computer Architecture",
        "credits": 3,
        "prerequisites": ["01:198:112"],
        "description": (
            "Introduction to computer systems, C programming, memory, and "
            "the software/hardware interface."
        ),
    },
    {
        "code": "01:198:344",
        "name": "Design and Analysis of Algorithms",
        "credits": 3,
        "prerequisites": ["01:198:112", "01:198:205", "01:198:206"],
        "description": (
            "Algorithm design paradigms (greedy, dynamic programming, divide and "
            "conquer) and rigorous analysis of correctness and efficiency."
        ),
    },
]

# Tool Function: Course Lookup
def get_course_info(course_name: str):
    """
    Return courses that match a name substring or code.
    """
    matches = []
    for course in COURSES:
        if course_name.lower() in course["name"].lower() \
           or course_name.lower() in course["code"].lower():
            matches.append(course)
    if not matches:
        return f"No courses found matching '{course_name}'."
    return matches

# Initialize OpenAIChatClient for GitHub Models
openai_chat_client = OpenAIChatClient(
    base_url= url,
    api_key= key,
    model_id= model
)

# Instantiate the ChatAgent
agent = ChatAgent(
    chat_client=openai_chat_client,
    instructions=(
        "You are a helpful AI Agent that helps students plan their Computer Science courses. "
        "You can call the 'get_course_info' tool to look up course details, prerequisites, "
        "credits, and descriptions."
    ),
    tools=[get_course_info]
)

# Interactive async loop
async def main():
    print("Welcome to the Rutgers CS Course Planner Agent!")
    print("Type 'quit' to exit.")
    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() == "quit":
            print("Exiting the course planner. ")
            break

        try:
            response = await agent.run(user_query)
            last_message = response.messages[-1]
            text_content = last_message.contents[0].text
            print("\n Course plan / suggestion:")
            print(text_content)
        except Exception as e:
            print(f" Error occurred: {e}")
            print("Try again or type 'quit' to exit.")

# Run the async main
if __name__ == "__main__":
    asyncio.run(main())

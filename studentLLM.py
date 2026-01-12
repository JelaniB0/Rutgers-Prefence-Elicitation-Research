"""
Initial thought was to use a student LLM agent to help with course preference elicitation research. 
The student LLM could be designed to simulate student behavior and preference, and would interact with agentic AI framework
that will utilize student responses and outside tools or data to provide insight into an accurate course recommendation system.

Things to be done: 
1. Define the scope and capabilities of the student LLM agent.
2. Define the interaction framework between the student LLM and other AI agents/tools.
3. Implement the student LLM agent using appropriate LLM technologies, decide what specific LLM model to use. 
"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from utils.azure_local import get_environment
from data import load_local_data

USE_AZURE, blob_service_client = get_environment()

if USE_AZURE:
    print("Fetching data from Azure")
else:
    print("Fetching data locally")
    data = load_local_data(os.getenv("LOCAL_DB_PATH", "local.db"))
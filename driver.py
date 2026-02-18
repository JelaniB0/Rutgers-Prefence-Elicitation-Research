"""
Driver for agent framework
"""

from agents.orchestrator_agent import OrchestratorAgent
from agents.shared_types import ConversationState
import asyncio
import json
import os
from datetime import datetime

"""added convesation_log json file functionality to log conversations for later analysis.
The log file stores conversations(user queries and agent responses), agents invoked, and timestamps for each turn in a conversation session. 

"""
log_file = "conversation_log.json"

def load_log() -> dict:
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return json.load(f)
    return {"sessions": []}

def save_log(log_data: dict):
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

async def main():
    """
    Function to run Orchestrator code
    """

    print("Rutgers CS Course Advisor - Orchestrator Agent")

    try:
        orchestrator = OrchestratorAgent()
        orchestrator.initialize_agents()

    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")

    print("Hello! I'm your Rutgers CS course advisor.")
    print("I am here to assist with course rankings and recommendations, please ask me about course recommendations.")
    print("Type 'quit' to exit.\n")

    state = ConversationState()

    log = load_log()
    session = {
        "session_id": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": datetime.now().isoformat(),
        "turns": []
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except(KeyboardInterrupt, EOFError):
            print("\n\nInterrupted. Exiting. ")
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Rutgers CS Course Advisor. Good luck with your courses!")
            break

        if not user_input:
            continue

        try:
            agents_invoked = []
            turns_start = datetime.now().isoformat()
            response, agents_invoked = await orchestrator.process_query(user_input, state)
            print(f"\nAgent Service: {response}\n")
            state.add_message("user", user_input)
            state.add_message("assistant", response)

            turn = {
                "timestamp": turns_start,
                "user_query": user_input,
                "agents_invoked": agents_invoked,
                "response": response
            }
            session["turns"].append(turn)
            log_to_save = {"sessions": log["sessions"] + [session]}
            save_log(log_to_save)

        except Exception as e:
            print(f"Error: An error occured while processing request: {str(e)}")

    session["ended_at"] = datetime.now().isoformat()
    log["sessions"].append(session)
    save_log(log)

if __name__ == "__main__":
    asyncio.run(main())

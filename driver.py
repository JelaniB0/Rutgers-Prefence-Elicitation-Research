"""
Driver for agent framework
"""

from agents.orchestrator_agent import OrchestratorAgent
from agents.parser_agent import ParserAgent
from agents.data_agent import DataAgent
import asyncio

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
            response = await orchestrator.process_query(user_input)
            print(f"\nAgent Service: {response}\n")

        except Exception as e:
            print(f"Error: An error occured while processing request: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

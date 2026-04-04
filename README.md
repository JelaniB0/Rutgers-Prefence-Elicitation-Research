# Multi-agent Computer Science Course Recommendation and Course Assistance Framework
The programs in this repository utilizes Microsoft's Agent SDK to allow the use of a multi-agent framework that can serve as a tool to Rutgers Computer Science students and academic advisors for course recommendation or course information retrieval queries. There are several agents used within the framework which includes the following: Parser, Orchestrator, Planning, Transcript, Constraint, and Constraint agent. The framework utilizes OpenAI models for all agent initializations and certain tools are provided to specific agents to better assist the agent with returning targeted, accurate, and satisfiable responses to Rutgers CS related course queries. 

### Collaborators
***Framework Development and Testing:***
Jelani Beaugris, Arpita Biswas, Aashay Sankhe

## Package Installation and Setup
- All packages needed to run the framework program are in the requirements.txt file. 

***1. Clone the repository***
```powershell
git clone https://github.com/JelaniB0/Rutgers-Prefence-Elicitation-Research.git
```

***2. Create and activate Python virtual environment***
'''
python -m venv venv
venv/Scripts/activate
'''

***3. Install required packages***
```
pip install -r requirements.txt
```

***4. Set up environment variables***
```powershell
Copy-Item .env.example .env
```
Then open `.env` and fill in your credentials, including github credentials and preferred openAI model used.

## Running Agent Framework 
The script '''driver.py''' initializes our agent framework, builds the agent DAG workflow graph, and enables queries to be logged to 
query_log.csv with further insight into what tools agents used while being called, what agents were called as a result of queries from users, the time the queries were written, etc (section has to be updated as query logger is to be updated soon)

To run, ensure that you've completed the package installation and setup step above. Once you've completed that, simply run driver.py in your terminal. 

1. Make sure you have all required packages installed 

2.  Running driver script: 
```powershell
python driver.py
```

3. All queries and agent responses are logged to query_log.csv. 





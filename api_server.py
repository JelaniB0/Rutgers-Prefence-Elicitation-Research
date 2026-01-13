from fastapi import FastAPI
from pydantic import BaseModel
from mcp_server import tools  # Import the module
from typing import List
import uvicorn
import json

app = FastAPI(title="Course Advisor MCP Server")

class AnalyzeRequest(BaseModel):
    completed: List[str]

@app.post("/tools/analyze_prerequisites")
def api_analyze_prerequisites(request: AnalyzeRequest):
    """Analyze prerequisites - calls FastMCP tool internally."""
    # Access the actual function from the FunctionTool wrapper
    return tools.analyze_prerequisites.fn(completed=request.completed)

@app.get("/resources/course_catalog/{school}")
def api_course_catalog(school: str):
    """Get course catalog - calls FastMCP resource internally."""
    # Access the actual function from the resource wrapper
    result = tools.load_course_catalog.fn(school=school)
    return json.loads(result)

@app.get("/")
def root():
    return {"status": "Course Advisor MCP Server Running", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
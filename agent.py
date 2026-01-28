# pip install -r requirements.txt
import os
import json
import asyncio
import numpy as np
from dotenv import load_dotenv
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions
from openai import OpenAI

# Load environment variables
load_dotenv()

url = os.environ.get("GITHUB_ENDPOINT", "https://models.inference.ai.azure.com")
key = os.environ["GITHUB_TOKEN"]
model = os.environ["GITHUB_MODEL_ID"]

content_safety_client = ContentSafetyClient(
    endpoint=os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_CONTENT_SAFETY_KEY"])
)

# OpenAI client for embeddings
embedding_client = OpenAI(
    base_url=url,
    api_key=key #uses github models compatible api
)

# ============================================================================
# RAG COMPONENTS
# ============================================================================

class CourseRAG:
    """
    RAG system for course information retrieval using vector embeddings.
    """
    def __init__(self, courses, cache_file="course_embeddings_cache.json"):
        self.courses = courses
        self.course_embeddings = []
        self.course_texts = []
        self.cache_file = cache_file
        
        # Try to load cached embeddings first
        if self._load_cached_embeddings():
            print(f" Loaded embeddings from cache ({len(self.courses)} courses)")
        else:
            # Create embeddings if cache doesn't exist
            print("Initializing RAG system with embeddings...")
            self._create_course_embeddings()
            self._save_embeddings_cache()
            print(f" RAG system ready with {len(self.courses)} courses")
    
    def _load_cached_embeddings(self):
        """
        Load embeddings from cache file if it exists.
        """
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Verify cache matches current courses
            if len(cache_data['embeddings']) != len(self.courses):
                print("Cache size mismatch, regenerating embeddings...")
                return False
            
            self.course_embeddings = cache_data['embeddings']
            self.course_texts = cache_data['texts']
            return True
            
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
    
    def _save_embeddings_cache(self):
        """
        Save embeddings to cache file for faster future loads.
        """
        try:
            cache_data = {
                'embeddings': self.course_embeddings,
                'texts': self.course_texts
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
            print(f" Embeddings cached to {self.cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _create_course_text(self, course):
        """
        Create a searchable text representation of a course.
        """
        text_parts = [
            f"Course Code: {course.get('code', 'N/A')}",
            f"Title: {course.get('title', 'N/A')}",
            f"Credits: {course.get('credits', 'N/A')}",
            f"Description: {course.get('description', 'N/A')}"
        ]
        
        # Add prerequisites if available
        if 'prerequisites' in course and course['prerequisites']:
            prereqs = ', '.join(course['prerequisites'])
            text_parts.append(f"Prerequisites: {prereqs}")
        
        return " | ".join(text_parts)
    
    def _create_course_embeddings(self):
        """
        Generate embeddings for all courses using OpenAI's embedding model.
        """
        print(f"Creating embeddings for {len(self.courses)} courses...")
        
        for i, course in enumerate(self.courses, 1):
            course_text = self._create_course_text(course)
            self.course_texts.append(course_text)
            
            # Progress indicator every 5 courses
            if i % 5 == 0 or i == len(self.courses):
                print(f" Progress: {i}/{len(self.courses)} courses processed...")
            
            try:
                # Get embedding from OpenAI-compatible API (GitHub Models)
                response = embedding_client.embeddings.create(
                    input=course_text,
                    model="text-embedding-3-small"  # GitHub Models supports this
                )
                embedding = response.data[0].embedding
                self.course_embeddings.append(embedding)
            except Exception as e:
                print(f"  Error creating embedding for {course.get('code', 'unknown')}: {e}")
                # Use zero vector as fallback
                self.course_embeddings.append([0.0] * 1536)
    
    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def retrieve_relevant_courses(self, query: str, top_k: int = 5):
        """
        Retrieve the most relevant courses based on semantic similarity.
        
        This is the core RAG RETRIEVAL step.
        """
        try:
            # Get embedding for the query
            response = embedding_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            
            # Calculate similarity scores
            similarities = []
            for i, course_embedding in enumerate(self.course_embeddings):
                similarity = self._cosine_similarity(query_embedding, course_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k most relevant courses
            relevant_courses = []
            for i, score in similarities[:top_k]:
                course = self.courses[i].copy()
                course['relevance_score'] = score
                relevant_courses.append(course)
            
            print(f"RAG retrieved {len(relevant_courses)} relevant courses for query: '{query}'")
            return relevant_courses
            
        except Exception as e:
            print(f"Error in RAG retrieval: {e}")
            return []

# ============================================================================
# INITIALIZE RAG SYSTEM
# ============================================================================

def load_courses_from_json(filepath: str = "rutgers_courses.json"):
    """
    Load courses from JSON file.
    """
    try:
        with open(filepath, 'r') as f:
            courses = json.load(f)
        print(f"Loaded {len(courses)} courses from {filepath}")
        return courses
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

# Load courses
COURSES = load_courses_from_json()

# Initialize RAG system
if COURSES:
    rag_system = CourseRAG(COURSES)
else:
    rag_system = None
    print("Warning: RAG system not initialized due to missing course data")

# ============================================================================
# RAG-ENHANCED TOOLS
# ============================================================================

def rag_search_courses(query: str, top_k: int = 5):
    """
    RAG-powered semantic search for courses.
    
    Uses vector embeddings to find courses semantically similar to the query,
    then returns the retrieved information for the LLM to use.
    """
    if not rag_system:
        return "RAG system not available. Please load course data."
    
    # RETRIEVAL step
    relevant_courses = rag_system.retrieve_relevant_courses(query, top_k=top_k)
    
    if not relevant_courses:
        return f"No courses found for query: '{query}'"
    
    # Format retrieved information for the LLM (AUGMENTATION step)
    result = f"Retrieved {len(relevant_courses)} most relevant courses for: '{query}'\n\n"
    
    for i, course in enumerate(relevant_courses, 1):
        result += f"{i}. {course.get('code', 'N/A')} - {course.get('title', 'N/A')}\n"
        result += f"   Credits: {course.get('credits', 'N/A')}\n"
        
        if 'prerequisites' in course and course['prerequisites']:
            prereqs = ', '.join(course['prerequisites'])
            result += f"   Prerequisites: {prereqs}\n"
        
        if 'description' in course:
            desc = course['description']
            if len(desc) > 200:
                desc = desc[:200] + "..."
            result += f"   Description: {desc}\n"
        
        result += f"   Relevance Score: {course.get('relevance_score', 0):.3f}\n\n"
    
    return result

def get_course_by_code(course_code: str):
    """
    Get detailed information about a specific course by its code.
    """
    if not COURSES:
        return "No courses available."
    
    course_code_clean = course_code.strip().upper()
    
    for course in COURSES:
        if course.get('code', '').upper() == course_code_clean:
            result = f"Course Details for {course.get('code')}:\n\n"
            result += f"Title: {course.get('title', 'N/A')}\n"
            result += f"Credits: {course.get('credits', 'N/A')}\n"
            
            if 'prerequisites' in course and course['prerequisites']:
                prereqs = ', '.join(course['prerequisites'])
                result += f"Prerequisites: {prereqs}\n"
            else:
                result += "Prerequisites: None\n"
            
            if 'description' in course:
                result += f"\nDescription: {course['description']}\n"
            
            if 'course_type' in course:
                result += f"Type: {course['course_type'].title()}\n"
            
            print(f"get_course_by_code tool used for: {course_code}")
            return result
    
    return f"Course {course_code} not found."

# ============================================================================
# VALIDATION 
# ============================================================================

def check_azure_content_safety(text: str, context: str = "input") -> tuple[bool, str]:
    try:
        request = AnalyzeTextOptions(text=text)
        response = content_safety_client.analyze_text(request)
        violations = []
        
        if hasattr(response, 'categories_analysis'):
            for category in response.categories_analysis:
                threshold = 1 if context == "output" else 2
                if category.severity > threshold:
                    violations.append(category.category)
        
        if violations:
            return False, f"Content contains harmful material: {', '.join(violations)}"
        return True, "Content is safe."
    except Exception as e:
        print(f"Azure safety check failed: {e}")
        return True, ""

async def check_with_llm_input(text: str, context: str, original_query: str = "") -> tuple[bool, str]:
    if context == "input":
        prompt = f"""You are a content policy classifier for an educational CS course planning assistant.

Analyze this user's input and respond only with these two options:
- "RELEVANT" if the input is related to planning Computer Science courses.
- "IRRELEVANT: [reason]" if the input is off-topic or inappropriate.

User input: "{text}"

Your response:"""
    else:
        prompt = f"""You are a quality control classifier for an educational CS course planning assistant.

Original user query: "{original_query}"
Agent's response: "{text}"

Respond only with:
- "VALID" if the response is appropriate and on-topic.
- "INVALID: [reason]" if the response has issues.

Your response:"""
    
    try:
        validation_agent = ChatAgent(
            chat_client=openai_chat_client,
            instructions=(
                f"You are a content policy classifier. "
                f"Respond ONLY with '{'RELEVANT' if context == 'input' else 'VALID'}' or "
                f"'{'IRRELEVANT' if context == 'input' else 'INVALID'}: [reason]'."
            ),
            tools=[]
        )
        
        response = await validation_agent.run(prompt)
        last_message = response.messages[-1]
        result = last_message.contents[0].text.strip().upper()
        
        if context == "input":
            if "RELEVANT" in result and "IRRELEVANT" not in result:
                return True, "Input is relevant."
            elif "IRRELEVANT" in result:
                reason = result.split(":", 1)[1].strip() if ":" in result else "Input is not related to course planning."
                return False, reason
            else:
                return False, "Unable to validate input."
        else:
            if "VALID" in result and "INVALID" not in result:
                return True, "Output is valid."
            elif "INVALID" in result:
                reason = result.split(":", 1)[1].strip() if ":" in result else "Response doesn't meet quality standards."
                return False, reason
            else:
                return False, "Response validation unclear."
    except Exception as e:
        print(f"LLM {context} validation failed: {e}")
        return False, "Validation service unavailable."

async def validate_user_input(text: str) -> tuple[bool, str]:
    is_safe, message = check_azure_content_safety(text, context="input")
    if not is_safe:
        return False, message
    
    is_relevant, message = await check_with_llm_input(text, context="input")
    if not is_relevant:
        return False, message
    
    return True, "Input is valid."

async def validate_agent_output(text: str, original_query: str) -> tuple[bool, str]:
    is_safe, message = check_azure_content_safety(text, context="output")
    if not is_safe:
        return False, message
    
    is_relevant, message = await check_with_llm_input(text, context="output", original_query=original_query)
    if not is_relevant:
        return False, message
    
    return True, "Output is valid."

# ============================================================================
# AGENT SETUP
# ============================================================================

openai_chat_client = OpenAIChatClient(
    base_url=url,
    api_key=key,
    model_id=model
)

# RAG-enhanced agent with semantic search
agent = ChatAgent(
    chat_client=openai_chat_client,
    instructions=(
        "You are a helpful AI assistant for Rutgers CS course planning.\n\n"
        "You have access to a RAG (Retrieval-Augmented Generation) system with these tools:\n"
        "1. 'rag_search_courses' - Semantic search using vector embeddings. Use this for general queries like "
        "'What courses are about AI?', 'Show me machine learning courses', 'Courses related to databases', etc.\n"
        "2. 'get_course_by_code' - Get specific course details by exact code (e.g., '01:198:344').\n\n"
        "The RAG system finds semantically similar courses even if exact keywords don't match. "
        "For example, searching 'neural networks' will find 'machine learning' courses.\n\n"
        "Always provide helpful information about prerequisites, credits, and course content. "
        "When courses have prerequisites, explain what students need to take first."
    ),
    tools=[rag_search_courses, get_course_by_code]
)

# ============================================================================
# MAIN LOOP
# ============================================================================

async def main():
    print("=" * 60)
    print("Rutgers CS Course Planner with RAG")
    print("=" * 60)
    
    if not COURSES:
        print("\n  Warning: No courses loaded.")
        return
    
    if not rag_system:
        print("\n  Warning: RAG system not initialized.")
        return
    
    print(f"\n Loaded {len(COURSES)} courses with RAG-powered semantic search")
    print("\nRAG enables semantic understanding - ask naturally:")
    print("  - 'What courses teach neural networks?'")
    print("  - 'I want to learn about databases'")
    print("  - 'Show me courses related to web development'")
    print("  - 'What are the AI courses?'")
    print("  - 'Tell me about 01:198:344'")
    print("\nType 'quit' to exit.\n")
    
    while True:
        user_query = input("Your question: ")
        if user_query.lower() == "quit":
            print("\n Exiting. ")
            break
        
        # Validate input
        is_valid_input, error_message = await validate_user_input(user_query)
        if not is_valid_input:
            print(f" Input validation failed: {error_message}\n")
            continue
        
        try:
            # Agent uses RAG to retrieve and generate response
            response = await agent.run(user_query)
            last_message = response.messages[-1]
            agent_output = last_message.contents[0].text
            
            # Validate output
            is_valid_output, error_message = await validate_agent_output(agent_output, user_query)
            if not is_valid_output:
                print(f" Output validation failed: {error_message}\n")
                continue
            
            print(f"\n{'='*60}")
            print("Course Assistant:")
            print(f"{'='*60}")
            print(agent_output)
            print()
            
        except Exception as e:
            print(f" Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    asyncio.run(main())

# checker for if tool is used during query/conversation
# look into guardrails for agent framework. 
# groundedness not available yet for ai agents with microsoft openai client. 
# intervention points available for user input, tool call, tool response, and output generation.
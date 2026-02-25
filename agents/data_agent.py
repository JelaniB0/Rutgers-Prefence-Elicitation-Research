"""
Data Agent code, RAG yet for semantic search.
"""

import os
import json
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from typing import Dict, List, Any
import re
from openai import AsyncOpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv


from .shared_types import AgentResponse, ConversationState

class DataAgent(ChatAgent):
    """
    Data agent that retrieves course information using semantic search and natural language understanding. Planning to expand 
    to resumes and other sources of data, plan to have agent extract data from external database. 
    """

    def __init__(self, client: OpenAIChatClient, model: str, courses_file: str = "rutgers_courses.json"):
        """
        Initialize data agent with RAG-based semantic search
        """

        super().__init__(
            chat_client=client,
            model=model,
            instructions=self._get_system_message()
        )

        self.model = model
        self.courses_file = courses_file
        self.courses_data=self._load_courses()

        self.embedding_client = AsyncOpenAI(
            api_key=os.environ["GITHUB_TOKEN"],
            base_url="https://models.inference.ai.azure.com/"
        )

        self.vector_db= self._initialize_vector_db()
        self._index_courses()

        print(f"[DataAgent] Initialized with model: {model}")
        print(f"[DataAgent] Loaded {len(self.courses_data)} courses")
        print(f"[DataAgent] Vector database initialized with GitHub Models embeddings")

    def _initialize_vector_db(self):
        """
        Initialize ChromaDB with Github model embeddings. 
        """

        client = chromadb.PersistentClient(path="./chroma_db")
        github_embedfunc = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["GITHUB_TOKEN"],
            api_base="https://models.inference.ai.azure.com/",
            model_name="text-embedding-3-small" # supported by git models
        )

        try:
            collection = client.get_collection(
                name = "rutgers_courses",
                embedding_function=github_embedfunc
            )
            print("[DataAgent] using existing embedding collection.")
        except:
            collection = client.create_collection(
                name="rutgers_courses",
                embedding_function=github_embedfunc,
                metadata={
                    "hnsw:space": "cosine", # don't know what this does necessarily
                    "description": "Rutgers CS courses with semantic search"
                }
            )
            print("[DataAgent] Created new collection")
        
        return collection

    def _index_courses(self):
        """Courses indexed into vector database"""
        if self.vector_db.count() > 0:
            print(f"[DataAgent] Vector DB already contains {self.vector_db.count()} courses")
            return
        
        print("[DataAgent] Indexing courses into vector database...")
        documents = []
        metadatas = []
        ids = []

        for course in self.courses_data:
            course_text = self._create_course_documents(course)

            documents.append(course_text)
            metadatas.append({
                "code": course.get('code', ''),
                "title": course.get('title', ''),
                "credits": str(course.get('credits', '3')),
                "level": self._extract_course_level(course.get('code', ''))
            })
            
            ids.append(course.get('code', f"course_{len(ids)}"))


        self.vector_db.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"[DataAgent] Indexed {len(documents)} courses")
    
    def _create_course_documents(self, course: Dict) -> str:
        """Create searchable text representation of course"""
        parts = [
            f"Course: {course.get('title', '')}",
            f"Code: {course.get('code', '')}",
            f"Description: {course.get('description', '')}",
        ]

        if 'prerequisites' in course:
            parts.append(f"Prerequisites: {course['prerequisites']}")

        if 'topics' in course:
            parts.append(f"Topics: {', '.join(course['topics'])}")

        return " ".join(parts)
    
    def _extract_course_level(self, course_code: str) -> str:
        """Extract course level from code"""
        match = re.search(r':(\d{3}):', course_code)
        if match:
            level = int(match.group(1))
            if level < 200:
                return "intro"
            elif level < 300:
                return "intermediate"
            else:
                return "advanced"
        return "unknown"

    def _get_system_message(self) -> str:
        
        return """You are a course data specialist for Rutgers University CS department.
        You receive pre-filtered relevant courses from a semantic search system.
        Your job is to:
        1. Analyze the retrieved courses for relevance to student needs
        2. Rank courses by how well they match student interests and goals
        3. Provide clear reasoning for each recommendation
        4. Consider prerequisites, difficulty level, and learning progression

        Always return your analysis in valid JSON format.
        """
    
    def _load_courses(self) -> List[Dict]:
        """Load courses from JSON file"""
        try:
            with open(self.courses_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'courses' in data:
                    return data['courses']
                return []
        except FileNotFoundError:
            print(f"[DataAgent] Error: {self.courses_file} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"[DataAgent] Error: Invalid JSON: {e}")
            return []
    
    async def fetch_courses(self, parsed_data: Dict, state: ConversationState) -> AgentResponse:
        """
        Retrieve courses using RAG-based semantic search.
        Arguments is parsed data from ParserAgent and current conversation state
        Returns AgentResponse with semantically matched courses.
        """

        try:
            entities = parsed_data.get('entities', {})
            intent = parsed_data.get('intent', 'course_recommendation')

            print(f"[DataAgent] Fetching courses for intent: {intent}")
            print(f"[DataAgent] Entities: {entities}")

            matched_courses = await self._rag_retrieve(entities, intent)

            return AgentResponse(
                success=True,
                data={ # data dictionary
                    'courses': matched_courses,
                    'total_found': len(matched_courses),
                    'search_method': 'semantic'
                },
                metadata={ # metadata dictionary
                    'search_criteria': entities,
                    'model_used': self.model
                }
            )
        except Exception as e:
            print(f"[DataAgent] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

            return AgentResponse(
                success=False,
                data=None,
                errors=[f"Data retrieval error: {str(e)}"]
            )
        
    async def _rag_retrieve(self, entities: Dict, intent: str) -> List[Dict]:
        """
        RAG-based semantic search using github models

        Args - Extracted entities (entities: interests, year, etc), intent - Query intent

        Returns list of matched courses with relevance scores
        """

        query = self._build_search_query(entities)
        print(f"[DataAgent] RAG query: {query}")
        
        # builds metadata filters -> look into this later
        # where_filter = self._build_metadata_filters(entities)
        where_filter = None

        # retrieve courses from vector DB
        retrieval_results = self.vector_db.query(
            query_texts=[query],
            n_results=20,
            where=where_filter if where_filter else None
        )

        retrieved_course_codes = retrieval_results['ids'][0]
        retrieved_distances = retrieval_results['distances'][0]

        retrieved_courses = []
        for i, code in enumerate(retrieved_course_codes):
            course = next((c for c in self.courses_data if c.get('code') == code), None)
            if course:
                course_copy = course.copy()
                # semantic similarity score for planninn agent
                course_copy['semantic_similarity'] = 1 - retrieved_distances[i]
                retrieved_courses.append(course_copy)

        print(f"[DataAgent] Retrieved {len(retrieved_courses)} courses")

        return retrieved_courses
    
    async def lookup_course(self, course_identifier: str, state: ConversationState) -> AgentResponse:
        """
        Look up a specific course by code or title. 

        Args:
        course_identifier: course code (e.g. CS:101) or title keyword (e.g. "Intro to CS") or partial title
        state: current conversation state for context

        Returns: 
        AgentResponse with course details if found, or error message if not found
        """

        try:
            print(f"[DataAgent] Looking up course: {course_identifier}")

            normalized = course_identifier.strip().lower()

            for course in self.courses_data:
                course_code = course.get('code', '').upper()

                normalized_search = normalized.replace(' ', '').replace(':', '')
                normalized_code = course_code.replace(' ', '').replace(':', '')

                number_only = re.search(r'\d+', normalized_search)

                if (normalized in course_code or normalized_search in normalized_code or (number_only and number_only.group() in normalized_code.split(':')[-1])):
                    print(f"[DataAgent] Found exact match: {course_code}")
                    return AgentResponse(
                        success=True,
                        data={
                            'course': course,
                            'lookup_method': 'exact_code_match'
                        },
                        metadata={
                            'search_query': course_identifier,
                            'model_used': self.model
                        }
                    )
            
            title_matches = []
            search_terms = [term for term in normalized.lower().split() if len(term) > 2]  # Ignore short words

            for course in self.courses_data:
                course_title = course.get('title', '').lower()

                if search_terms and all(term in course_title for term in search_terms):
                    title_matches.append(course)
            
            if len(title_matches) == 1:
                print(f"[DataAgent] Found title match: {title_matches[0].get('code', '')}")
                return AgentResponse(
                    success=True,
                    data={
                        'course': title_matches[0],
                        'lookup_method': 'title_keyword_match'
                    },
                    metadata={
                        'search_query': course_identifier,
                        'model_used': self.model
                    }
                )
            
            elif len(title_matches) > 1:
                print(f"[DataAgent] Found {len(title_matches)} title matches")
                return AgentResponse(
                    success=True,
                    data={
                        'courses': title_matches,
                        'lookup_method': 'title_keyword_match_multiple',
                        'needs_disambiguation': True
                    },
                    metadata={
                        'search_query': course_identifier,
                        'model_used': self.model,
                        'matches_found': len(title_matches)
                    }
                )
            
            print(f"[DataAgent] No matches found for: {course_identifier}")
            return AgentResponse(
                success=False,
                data={'attempted_search': 'course_identifier'},
                errors=[f"No course found matching '{course_identifier}'"]
            )
        
        except Exception as e:
            print(f"[DataAgent] ERROR during course lookup: {str(e)}")
            import traceback
            traceback.print_exc()

            return AgentResponse(
                success=False,
                data=None,
                errors=[f"Course lookup error: {str(e)}"]
            )
              
    def _build_search_query(self, entities: Dict) -> str:
        """Builds natural language query for vector search"""
        query_parts = []

        if entities.get('interests'):
            interests = entities['interests']
            query_parts.append(f"courses about {' ,'.join(interests)}")

        if entities.get('career_path'):
            query_parts.append(f"relevant to {entities['career_path']} career")
        
        if entities.get('specific_courses'):
            query_parts.append(f"similar to {', '.join(entities['specific_courses'])}")
        
        if entities.get('difficulty_preference'):
            query_parts.append(f"{entities['difficulty_preference']} level")
        
        return " ".join(query_parts) if query_parts else "computer science courses"
    
    # def _build_metadata_filters(self, entities: Dict) -> Dict:
    #     """Build metadata filters for vector search (optional pre-filtering)"""
    #     filters = {}
        
    #     # Optional: Filter by course level based on year
    #     year = entities.get('year')
    #     if year in ['freshman', 'sophomore']:
    #         filters['level'] = {'$in': ['intro', 'intermediate']}
    #     elif year in ['junior', 'senior']:
    #         filters['level'] = {'$in': ['intermediate', 'advanced']}
        
    #     return filters if filters else None
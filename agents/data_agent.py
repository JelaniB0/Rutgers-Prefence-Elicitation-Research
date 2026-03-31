"""
Data Agent code, RAG yet for semantic search.
"""

import os
import json
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from typing import Dict, List
import re
from openai import AsyncOpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import httpx
import time
from datetime import date

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

        # request and cache management for SOC for further course filtering based on offered and non-offered courses. 
        self.soc_cache = {} # term year as key
        self.RUTGERS_SOC_API = "https://classes.rutgers.edu/soc/api/courses.json"
        self.CACHE_NEXT = 60 * 60 * 12 # recache course data every 

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
                    courses = data
                elif isinstance(data, dict) and 'courses' in data:
                    courses = data['courses']
                else:
                    courses = []
                # Build lookup map once at load time
                self.code_to_title = {c["code"]: c["title"] for c in courses}
                return courses
        except FileNotFoundError:
            print(f"[DataAgent] Error: {self.courses_file} not found")
            self.code_to_title = {}
            return []
        except json.JSONDecodeError as e:
            print(f"[DataAgent] Error: Invalid JSON: {e}")
            self.code_to_title = {}
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

            # filter out courses already taken or in progress based on transcript data in conversation state. 
            if state.transcript_data:
                completed = {c["code"] for c in state.transcript_data.get("completed_courses", [])}
                in_progress = {c["code"] for c in state.transcript_data.get("in_progress_courses", [])}
                exclude = completed | in_progress

                before = len(matched_courses)
                matched_courses = [c for c in matched_courses if c.get("code") not in exclude]
                print(f"[DataAgent] Filtered {before - len(matched_courses)} already-taken courses, {len(matched_courses)} remaining")

            # filter out courses not offered in upcoming semester based on live SOC data. 
            if state.resolved_semester:
                semester = state.resolved_semester
            else:
                semester = self.resolve_semester(state.user_query or "")
                # only store if query had explicit semester mention
                if any(k in (state.user_query or "").lower() for k in ["next", "spring", "fall", "summer", "winter", "this sem"]):
                    state.resolved_semester = semester
                    
            offered = await self._fetch_soc_courses(semester)
            if offered is not None and len(offered) > 0:
                before = len(matched_courses)
                matched_courses = [
                    c for c in matched_courses
                    if c.get("code", "").split(":")[-1].strip() in offered
                ]
                print(f"[DataAgent] Filtered {before - len(matched_courses)} unoffered courses, {len(matched_courses)} remaining")
            elif offered is not None and len(offered) == 0:
                print(f"[DataAgent] SOC returned no courses for {semester} — schedule may not be posted yet, using fallback")
            else:
                print(f"[DataAgent] SOC API unavailable — using full course list as fallback")

            return AgentResponse(
                success=True,
                data={ # data dictionary
                    'courses': matched_courses,
                    'total_found': len(matched_courses),
                    'search_method': 'semantic',
                    'semester': semester
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

        # rewrites query to be more effective for semantic search. 
        try:
            improved_query = await self.run(f"""
            Rewrite this into a strong semantic search query for course retrieval.

            Original query:
            {query}

            Student intent:
            {entities}

            Focus on topics, skills, and career relevance.
            """)
            query = improved_query.content if hasattr(improved_query, "content") else str(improved_query)
        except Exception as e:
            print(f"[DataAgent] Query rewrite failed, using original: {e}")
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
                retrieved_courses.append(self._enrich_course(course_copy))

        print(f"[DataAgent] Retrieved {len(retrieved_courses)} courses")

        # LLM filtering to remove weakly relevant courses based on student needs and course details. 

        try:
            filter_prompt = f"""
            From these courses, remove ones that are weakly relevant.

            Student:
            {entities}

            Courses:
            {json.dumps(retrieved_courses, indent=2)}

            Return JSON ONLY in this format:

            {{ "keep": ["01:198:111", "01:198:205"] }}
            """

            try:
                response = await self.run(filter_prompt)
                response_text = response.messages[-1].contents[0].text

                # extract JSON from text using regex (robust in case LLM adds extra text)
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                
                if json_match:
                    parsed = json.loads(json_match.group())
                    keep_codes = parsed.get("keep", [])
                    
                    # filter the original courses
                    if keep_codes:
                        retrieved_courses = [
                            c for c in retrieved_courses if c.get("code") in keep_codes
                        ]
                        print(f"[DataAgent] Filtered to {len(retrieved_courses)} relevant courses")
                else:
                    print("[DataAgent] No JSON found in LLM output, keeping original list")

            except Exception as e:
                print(f"[DataAgent] LLM filtering failed, keeping original list: {e}")

        except Exception as e:
            print(f"[DataAgent] LLM filtering failed, keeping original list: {e}")

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

            #Exact code match first 
            for course in self.courses_data:
                course_code = course.get('code', '').upper()
                normalized_search = normalized.replace(' ', '').replace(':', '')
                normalized_code = course_code.replace(' ', '').replace(':', '')
                number_only = re.search(r'\d+', normalized_search)

                if (normalized in course_code or normalized_search in normalized_code or
                    (number_only and number_only.group() in normalized_code.split(':')[-1])):
                    print(f"[DataAgent] Found exact match: {course_code}")
                    return AgentResponse(
                        success=True,
                        data={'course': course, 'lookup_method': 'exact_code_match'},
                        metadata={'search_query': course_identifier, 'model_used': self.model}
                    )

            # Semantic search via vector DB
            results = self.vector_db.query(
                query_texts=[course_identifier],
                n_results=3
            )

            codes = results['ids'][0]
            distances = results['distances'][0]

            # Strong single match
            if codes and distances[0] < 0.5:
                course = next((c for c in self.courses_data if c.get('code') == codes[0]), None)
                if course:
                    print(f"[DataAgent] Found semantic match: {codes[0]} (distance: {distances[0]:.3f})")
                    return AgentResponse(
                        success=True,
                        data={'course': self._enrich_course(course), 'lookup_method': 'semantic_match'},
                        metadata={'search_query': course_identifier, 'model_used': self.model}
                    )

            # Multiple close matches — disambiguate choices. 
            close_matches = [
                next((c for c in self.courses_data if c.get('code') == code), None)
                for code, dist in zip(codes, distances) if dist < 0.45
            ]
            close_matches = [c for c in close_matches if c]

            if len(close_matches) > 1:
                return AgentResponse(
                    success=True,
                    data={'courses': close_matches, 'needs_disambiguation': True},
                    metadata={'search_query': course_identifier, 'model_used': self.model}
                )

            print(f"[DataAgent] No matches found for: {course_identifier}")
            return AgentResponse(
                success=False,
                data={'attempted_search': course_identifier},
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
    
    # Course Data Preprocessing

    def _extract_prereqs(self, description: str) -> tuple[str, str]:
        """Split prereq sentence from the rest of the description."""
        match = re.match(r'(Prerequisites?:.*?\.)\s*(.*)', description, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", description

    def _resolve_codes(self, prereq_text: str) -> str:
        """Replace bare course codes with 'CODE (Title)' using the loaded map."""
        for code, title in self.code_to_title.items():
            if code in prereq_text:
                prereq_text = prereq_text.replace(code, f"{code} ({title})")
        return prereq_text

    def _enrich_course(self, course: dict) -> dict:
        """Extract and resolve prerequisites, clean up description."""
        course = course.copy()
        prereq_text, clean_desc = self._extract_prereqs(course.get("description", ""))
        course["prerequisites"] = self._resolve_codes(prereq_text)
        course["description"] = clean_desc
        return course
    
    async def _fetch_soc_courses(self, semester: dict) -> set[str]:
        """
        Fetcjes live offered CS course numbers from Rutgers SOC API for given semester. Caches results to avoid excessive requests and slower runtime due to collecting offered course data.
        Each run. 
        """
        now = time.time()
        cache_key = f"{semester['term']}_{semester['year']}"  # e.g. "9_2026"

        # cache hit — same semester and not stale
        cached = self.soc_cache.get(cache_key)
        if cached and (now - cached["fetched_at"]) < self.CACHE_NEXT:
            print(f"[DataAgent] Using cached SOC data for {cache_key}")
            return cached["courses"]

        print("[DataAgent] Fetching live SOC data from Rutgers API...")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(self.RUTGERS_SOC_API, params = {
                    "year": semester.get("year"),
                    "term":semester["term"],
                    "campus": "NB", # keep to rutgers nb only for now.
                })
                resp.raise_for_status()
                all_courses = resp.json()

                offered = {
                    str(c["courseNumber"]) for c in all_courses if str(c.get("subject", "")) == "198" and c.get("level") == "U"
                }

                self.soc_cache[cache_key] = {"courses": offered, "fetched_at": now}
                print(f"[DataAgent] Cached {len(offered)} offered CS courses for {cache_key}")
                return offered

        except Exception as e:
            print(f"[DataAgent] SOC API unavailable: {e} — falling back to full course list")
            return None  # signals caller to skip the filter

    TERM_MAP = {
        "spring": 1,
        "summer": 7,
        "fall": 9,
        "autumn": 9,
        "winter": 0,
    }

    def resolve_semester(self, user_query: str) -> dict:
        query = user_query.lower()
        today = date.today()
        month, year = today.month, today.year

        import re
        year_match = re.search(r"20\d{2}", query)
        explicit_year = int(year_match.group()) if year_match else None

        if "spring" in query:
            term, y = 1, explicit_year or (year + 1 if month >= 9 else year)
        elif "summer" in query:
            term, y = 7, explicit_year or year
        elif "winter" in query:
            term, y = 0, explicit_year or (year + 1 if month >= 9 else year)
        elif "fall" in query or "autumn" in query:
            term, y = 9, explicit_year or (year + 1 if month >= 9 else year)
        elif "next semester" in query or "next sem" in query:
            if month <= 5:   term, y = 9, year       # spring → Fall
            elif month <= 8: term, y = 9, year       # summer → Fall
            elif month <= 11: term, y = 1, year + 1  # fall → Spring
            else:            term, y = 1, year + 1   # winter → Spring
        elif "this semester" in query or "current" in query:
            if month <= 1:   term, y = 0, year       # January → Winter
            elif month <= 5: term, y = 1, year       # Feb–May → Spring
            elif month <= 8: term, y = 7, year       # Jun–Aug → Summer
            else:            term, y = 9, year       # Sep–Dec → Fall
        else:
            # default to current semester
            if month <= 1:   term, y = 0, year
            elif month <= 5: term, y = 1, year
            elif month <= 8: term, y = 7, year
            else:            term, y = 9, year

        return {"term": term, "year": y}

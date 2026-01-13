from dataclasses import dataclass
from typing import List

@dataclass
class Course:
    course_id: str
    name: str

@dataclass
class StudentProfile:
    school: str
    completed_courses: List[Course]
def load_local_data(school: str):
    return {
        "courses": [
            {"course_id": "CS101", "prerequisites": []},
            {"course_id": "CS102", "prerequisites": ["CS101"]},
            {"course_id": "CS103", "prerequisites": ["CS101", "CS102"]},
        ]
    }
def load_azure_data(blob_service_client, school: str):
    print("Loading course data from Azure Blob Storage...")
    # Real logic later
    return {
        "courses": [
            {"course_id": "CS101", "prerequisites": []},
            {"course_id": "CS102", "prerequisites": ["CS101"]},
        ]
    }
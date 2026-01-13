from utils.azure_env import get_environment
from data.local import load_local_data
from data.azure import load_azure_data

def load_data(school: str):
    use_azure, blob_client = get_environment()

    if use_azure:
        return load_azure_data(blob_client, school)

    return load_local_data(school)
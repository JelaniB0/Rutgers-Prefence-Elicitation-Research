import os
from dotenv import load_dotenv

load_dotenv()  # make sure .env variables are available

def get_environment():
    """
    Determines environment: Azure (RBAC login) or local fallback.
    Returns: USE_AZURE (bool), blob_service_client or None
    """
    ENV = os.getenv("ENV", "local")
    try:
        if ENV == "azure_login":
            from azure.identity import AzureCliCredential
            from azure.storage.blob import BlobServiceClient

            account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
            if not account_name:
                raise ValueError("AZURE_STORAGE_ACCOUNT_NAME not set for login mode!")

            account_url = f"https://{account_name}.blob.core.windows.net"
            credential = AzureCliCredential()  # explicitly use CLI login
            blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

            print("Connected to Azure Blob Storage (RBAC login)")
            return True, blob_service_client

        else:
            # fallback to local
            raise ValueError("Forcing local mode")

    except Exception as e:
        print(f"Azure not available: {e}")
        return False, None
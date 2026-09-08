"""Shared Azure OpenAI configuration for the latest hub-and-spoke workflow."""

from dataclasses import asdict, dataclass, field, fields
import os
from pathlib import Path

from agent_framework.openai import OpenAIResponsesClient
from dotenv import load_dotenv
from openai import AsyncOpenAI
if __package__:
    from .metered_client import MeteredResponsesClient
else:
    # Standalone DAG construction does not run a query/session.
    MeteredResponsesClient = OpenAIResponsesClient


@dataclass(frozen=True)
class ModelConfig:
    """Explicit Azure deployment assignments for the seven LLM-backed roles."""

    parser: str
    orchestrator: str
    data: str
    constraint: str
    planning: str
    transcript: str
    response_writer: str

    @classmethod
    def from_environment(cls):
        return cls(**{f.name: _required_environment_value(
            f"AZURE_OPENAI_{f.name.upper()}_DEPLOYMENT") for f in fields(cls)})

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AzureOpenAISettings:
    """Environment-backed settings used by chat and embedding clients."""

    endpoint: str
    api_key: str = field(repr=False)
    embedding_deployment: str
    models: ModelConfig


def _required_environment_value(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill in your Azure OpenAI settings."
        )
    return value


def get_azure_openai_settings() -> AzureOpenAISettings:
    """Load and validate the Azure OpenAI v1 settings without exposing secrets."""

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    endpoint = _required_environment_value("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if "/api/projects/" in endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT must be an OpenAI-compatible /openai/v1 endpoint, "
            "not the Azure AI project endpoint."
        )
    if not endpoint.endswith("/openai/v1"):
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT must end with /openai/v1 (a trailing slash is also accepted)."
        )

    return AzureOpenAISettings(
        endpoint=endpoint,
        api_key=_required_environment_value("AZURE_OPENAI_API_KEY"),
        embedding_deployment=(
            os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small").strip()
            or "text-embedding-3-small"
        ),
        models=ModelConfig.from_environment(),
    )


def create_agent_client(
    settings: AzureOpenAISettings | None = None,
) -> OpenAIResponsesClient:
    """Create the Responses API client used by all Agent Framework agents."""

    settings = settings or get_azure_openai_settings()
    return MeteredResponsesClient(
        base_url=settings.endpoint,
        api_key=settings.api_key,
        model_id=settings.models.orchestrator,
    )


def create_async_openai_client(
    settings: AzureOpenAISettings | None = None,
) -> AsyncOpenAI:
    """Create a raw async client for non-Agent-Framework Responses API calls."""

    settings = settings or get_azure_openai_settings()
    return AsyncOpenAI(
        base_url=settings.endpoint,
        api_key=settings.api_key,
    )

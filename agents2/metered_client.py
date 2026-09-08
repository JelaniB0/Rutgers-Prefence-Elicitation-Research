"""Meter actual responses before Agent Framework parses them or agents use them."""

from agent_framework.openai import OpenAIResponsesClient
from .inference_metrics import record_response


class MeteredResponsesClient(OpenAIResponsesClient):
    def _parse_response_from_openai(self, response, options):
        record_response(response, options.get("model_id") or self.model_id)
        return super()._parse_response_from_openai(response, options=options)

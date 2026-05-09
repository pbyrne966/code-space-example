"""Mock Ollama client for ingestion and RAG tests."""

from src.db_service.schemas import MAX_EMBEDDING_DIMENSION
from src.model_service.models import ModelClient, ModelConfig, ModelOutput


class MockOllamaClient(ModelClient):
    """Deterministic test double that mimics the Ollama client surface."""

    def __init__(
        self,
        model_name: str = "mock-qwen",
        chat_output: str = (
            '{"answer":"mock answer","citations":[],"calculation_program":null,'
            '"requery":null}'
        ),
    ) -> None:
        self._config = ModelConfig(
            base_url="http://example.com",
            model_name=model_name,
            chat_endpoint="/chat",
            max_tokens=32,
            allowed_timeout=5,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )
        self.embedded_texts: list[str] = []
        self.prompts: list[str] = []
        self.response_formats: list[dict | str | None] = []
        self.chat_output = chat_output

    def server_alive(self) -> bool:
        """Pretend the model server is always reachable."""
        return True

    def model_exists(self) -> bool:
        """Pretend the model already exists locally."""
        return True

    def wait_until_model_ready(self) -> bool:
        """Pretend the model is always ready."""
        return True

    def initialize(self) -> None:
        """No-op initialization for the test double."""

    def get_config(self) -> ModelConfig:
        return self._config

    def embed(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return [
            float(len(text)),
            float(len(text) % 10),
            *([1.0] * (MAX_EMBEDDING_DIMENSION - 2)),
        ]

    def query_batch(self, prompts: list[str]) -> list[ModelOutput]:
        """Run the mock single-query behavior across a prompt batch."""
        return [self.query_single(prompt) for prompt in prompts]

    def query_single(
        self,
        prompt: str,
        http_method: str = "POST",
        response_format: dict | str | None = None,
    ) -> ModelOutput:
        self.prompts.append(prompt)
        self.response_formats.append(response_format)
        raw_response = {"message": {"content": self.chat_output}}
        return ModelOutput(
            request_id="mock-request",
            prompt=prompt,
            output=self.chat_output,
            raw_response=raw_response,
            tokens_used=0,
        )

import time
import tomllib
import uuid
from pathlib import Path
from typing import Any, Protocol

import requests
from pydantic import BaseModel, Field, model_validator

from src.logger import get_logger
from src.utils.http_utils import serialize_response, supported_http_method

logger = get_logger("model_download")


class ModelConfig(BaseModel):
    base_url: str
    model_name: str
    model_embed: str | None = Field(default=None)
    chat_endpoint: str
    batch_endpoint: str | None = None
    max_tokens: int
    allowed_timeout: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 42

    @model_validator(mode="after")
    def default_embedding_model(self) -> "ModelConfig":
        if self.model_embed is None:
            self.model_embed = self.model_name
        return self

    @classmethod
    def load_from_toml(cls, path: Path) -> "ModelConfig":
        logger.info("Loading model config from %s", path)

        with path.open("rb") as f:
            data = tomllib.load(f)

        return cls(**data)


class ModelInput(BaseModel):
    request_id: str
    prompt: str


class ModelOutput(BaseModel):
    request_id: str
    prompt: str
    output: str
    raw_response: dict[str, Any] | None = None


class ModelClient(Protocol):
    def server_alive(self) -> bool: ...
    def model_exists(self) -> bool: ...
    def wait_until_model_ready(self) -> bool: ...
    def initialize(self) -> None: ...
    def query_single(
        self,
        prompt: str,
        http_method: str = "POST",
        response_format: dict[str, Any] | str | None = None,
    ) -> ModelOutput: ...
    def query_batch(self, prompts: list[str]) -> list[ModelOutput]: ...
    def embed(self, text: str) -> list[float]: ...
    def get_config(self) -> ModelConfig: ...


class OllamaQwenClient:
    def __init__(self, model_config: ModelConfig) -> None:
        self.config = model_config
        self.is_model_ready = False

    def initialize(self) -> None:
        logger.info(
            "Initializing model client for model=%s embed_model=%s",
            self.config.model_name,
            self.config.model_embed,
        )

        if not self.server_alive():
            logger.error("Model server is not reachable at %s", self.config.base_url)
            raise RuntimeError("Model server is not reachable")

        logger.info("Model server is alive")

        if not self.model_exists():
            self.fetch_model()
        else:
            logger.info("Model already exists locally: %s", self.config.model_name)

        if not self.wait_until_model_ready():
            logger.error("Model failed to become ready: %s", self.config.model_name)
            raise RuntimeError("Model failed to become ready")

        logger.info("Model is ready: %s", self.config.model_name)

    def model_exists(self) -> bool:
        url = f"{self.config.base_url.rstrip('/')}/api/tags"

        try:
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            data = response.json()

            self.does_model_exist = any(
                model.get("name") == self.config.model_name
                for model in data.get("models", [])
            )
            return self.does_model_exist

        except requests.RequestException as exc:
            logger.warning("Failed checking model existence: %s", exc)
            return False

    def fetch_model(self) -> None:
        url = f"{self.config.base_url.rstrip('/')}/api/pull"

        logger.info("Ensuring model is available: %s", self.config.model_name)

        try:
            response = requests.post(
                url,
                json={"name": self.config.model_name},
                timeout=600,
            )
            response.raise_for_status()
            logger.info("Model fetch/pull completed for %s", self.config.model_name)
            self.does_model_exist = True

        except requests.RequestException as exc:
            logger.warning(
                "Model fetch failed for %s. Readiness check will verify "
                "availability. Error: %s",
                self.config.model_name,
                exc,
            )
            raise RuntimeError("Failed to fetch model") from exc

    def server_alive(self, retries: int = 5, delay: float = 1.0) -> bool:
        url = f"{self.config.base_url.rstrip('/')}/api/tags"

        for attempt in range(1, retries + 1):
            logger.info("Checking model server health, attempt %s/%s", attempt, retries)

            try:
                response = requests.get(url, timeout=3)
                if response.ok:
                    return True

                logger.warning(
                    "Server health check failed with status=%s body=%s",
                    response.status_code,
                    response.text,
                )

            except requests.RequestException as exc:
                logger.warning("Server health check error: %s", exc)

            time.sleep(delay)

        return False

    def wait_until_model_ready(self, retries: int = 10, delay: float = 2.0) -> bool:
        for attempt in range(1, retries + 1):
            logger.info("Checking model readiness, attempt %s/%s", attempt, retries)

            try:
                result = self.query_single("Return only the word ready.", "POST")
                self.is_model_ready = bool(result.output.strip())

                if self.is_model_ready:
                    return True

            except Exception as exc:
                logger.warning("Model readiness check failed: %s", exc)

            time.sleep(delay)
        self.is_model_ready = False
        return False

    def _extract_output(self, data: dict[str, Any]) -> str:
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        raise ValueError("Unsupported model response shape")

    def build_payload(
        self,
        prompt: str,
        response_format: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_predict": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        if self.config.seed is not None:
            options["seed"] = self.config.seed

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "options": options,
            "stream": False,
        }
        if response_format is not None:
            payload["format"] = response_format
        return payload

    def query_single(
        self,
        prompt: str,
        http_method: str = "POST",
        response_format: dict[str, Any] | str | None = None,
    ) -> ModelOutput:
        request_id = str(uuid.uuid4())
        method = supported_http_method(http_method)

        logger.info("Sending single model request request_id=%s", request_id)

        return self.send_request(
            ModelInput(prompt=prompt, request_id=request_id),
            method,
            response_format=response_format,
        )

    def send_request(
        self,
        model_input: ModelInput,
        http_method: str,
        response_format: dict[str, Any] | str | None = None,
    ) -> ModelOutput:
        query_url = (
            f"{self.config.base_url.rstrip('/')}/"
            f"{self.config.chat_endpoint.lstrip('/')}"
        )

        response = requests.request(
            method=http_method,
            url=query_url,
            json=self.build_payload(model_input.prompt, response_format),
            timeout=self.config.allowed_timeout,
        )

        data = serialize_response(response)

        output = self._extract_output(data)

        logger.info("Model request completed request_id=%s", model_input.request_id)

        return ModelOutput(
            request_id=model_input.request_id,
            prompt=model_input.prompt,
            output=output,
            raw_response=data,
        )

    def query_batch(self, prompts: list[str]) -> list[ModelOutput]:
        logger.info("Sending fake batch request batch_size=%s", len(prompts))

        return [self.query_single(prompt, "POST") for prompt in prompts]

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.config.base_url}/api/embeddings",
            json={
                "model": self.config.model_embed,
                "prompt": text,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def get_config(self) -> ModelConfig:
        return self.config


class BaseModelFactory:
    @classmethod
    def create(cls, model_type: str, model_config_path: Path) -> ModelClient:
        model_type = model_type.upper()

        logger.info("Creating model client model_type=%s", model_type)

        if model_type == "QWEN":
            model_config = ModelConfig.load_from_toml(model_config_path)
            client = OllamaQwenClient(model_config)
            client.initialize()
            return client

        logger.error("Unsupported model type: %s", model_type)
        raise ValueError(f"Unsupported model type: {model_type}")

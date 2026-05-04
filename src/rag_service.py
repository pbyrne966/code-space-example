import hashlib
import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.db_service.schemas import RetrievalChunkTable
from src.model_service.models import ModelClient


class RagAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class RetrievalResultRow(Protocol):
    RetrievalChunkTable: RetrievalChunkTable
    distance: float


class RetrieverClient(Protocol):
    def retrieve(self, query: str) -> list[RetrievalResultRow]: ...


class RAGCache:
    def __init__(self):
        self._cache: dict[str, RagAnswer] = {}

    def get(self, key: str) -> RagAnswer | None:
        return self._cache.get(key)

    def set(self, key: str, value: RagAnswer) -> None:
        self._cache[key] = value


class RAGQwenService:
    def __init__(self, model_client: ModelClient, retriever: RetrieverClient):
        self.model_client = model_client
        self.retriever = retriever
        self.model_config = self.model_client.get_config()
        self.cache = RAGCache()

    def _cache_key(self, question: str) -> str:
        payload = {
            "question": " ".join(question.strip().split()),
            "model": self.model_config.model_name,
            "max_tokens": self.model_config.max_tokens,
            "temperature": self.model_config.temperature,
            "top_p": self.model_config.top_p,
            "seed": self.model_config.seed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def build_context_block(
        self, index: int, chunk: RetrievalChunkTable, distance: float
    ) -> str:
        return f"""
        [Source {index}]
        chunk_id: {chunk.chunk_id}
        record_id: {chunk.record_id}
        metric: {chunk.metric}
        years: {chunk.years}
        period_labels: {chunk.period_labels}
        distance: {distance}

        {chunk.text}
        """.strip()

    def build_prompt(self, question: str, context: list[str]) -> str:
        context_string = "\n\n---\n\n".join(context)
        return f"""
You are answering using only the provided retrieved context.

Rules:
- Output a single JSON object only.
- Do not wrap the output in markdown or extra prose.
- If the answer is not in the context, set answer to "I don't know".
- Do not invent numbers.
- Return exactly these keys: answer, citations.
- citations must be a JSON array of chunk_id values used to support the answer.
- Be concise and precise.

Question:
{question}

Retrieved context:
{context_string}

Return JSON in this exact shape:
{{"answer":"...","citations":["chunk_id_1","chunk_id_2"]}}
""".strip()

    def _parse_answer(self, output: str) -> RagAnswer:
        try:
            return RagAnswer.model_validate_json(output)
        except ValidationError as exc:
            raise ValueError(
                f"Model output did not match RagAnswer schema: {exc}"
            ) from exc

    def answer(self, question: str) -> RagAnswer:
        cache_key = self._cache_key(question)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        results = self.retriever.retrieve(question)
        context_blocks = []

        for i, row in enumerate(results, start=1):
            chunk = row.RetrievalChunkTable
            distance = row.distance
            context_blocks.append(self.build_context_block(i, chunk, distance))

        prompt = self.build_prompt(question, context_blocks)
        model_output = self.model_client.query_single(prompt)
        parsed_output = self._parse_answer(model_output.output)
        self.cache.set(cache_key, parsed_output)
        return parsed_output

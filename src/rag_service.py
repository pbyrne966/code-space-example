from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.chunking_service.data_types import RetrievalChunk
from src.db_service.postgres_controllers import PostgresChunkStore
from src.model_service.models import ModelClient


class RagAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class RAGService:
    def __init__(self, model_client: ModelClient, retriever: PostgresChunkStore):
        self.model_client = model_client
        self.retriever = retriever
        self.model_config = self.model_client.get_config()

    def build_context_block(
        self, index: int, chunk: RetrievalChunk, distance: float
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

Return JSON following this type:
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
        results = self.retriever.retrieve(question)
        context_blocks = []

        for i, row in enumerate(results, start=1):
            chunk = row.chunk
            distance = row.distance
            context_blocks.append(self.build_context_block(i, chunk, distance))

        prompt = self.build_prompt(question, context_blocks)
        model_output = self.model_client.query_single(prompt)
        return self._parse_answer(model_output.output)

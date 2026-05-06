import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.aggregator_service.executor import execute_calculation_program
from src.aggregator_service.query_intent import (
    CalculationProgram,
    TableValueCandidate,
)
from src.chunking_service.period_extraction import extract_period_data
from src.data_types import RetrievalChunk, RetrievedChunkRecord
from src.db_service.postgres_controllers import PostgresChunkStore
from src.logger import get_logger
from src.model_service.models import ModelClient

logger = get_logger("rag_service")


class RawRagAnswer(BaseModel):
    """Raw structured response returned by the model."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    calculation_program: CalculationProgram | None


class RagAnswer(BaseModel):
    """Final scalar answer with supporting evidence and optional computation trace."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    turn_program: str | None = None


class RAGService:
    def __init__(self, model_client: ModelClient, retriever: PostgresChunkStore):
        self.model_client = model_client
        self.retriever = retriever
        self.model_config = self.model_client.get_config()

    def build_context_block(
        self, index: int, chunk: RetrievalChunk, distance: float
    ) -> str:
        table_values = [value.model_dump() for value in chunk.table_values]
        return f"""
[Source {index}]
chunk_id: {chunk.chunk_id}
record_id: {chunk.record_id}
metric: {chunk.metric}
years: {chunk.years}
period_labels: {chunk.period_labels}
distance: {distance}
table_values: {json.dumps(table_values)}

{chunk.text}
""".strip()

    def build_table_value_candidates(
        self, results: list[RetrievedChunkRecord]
    ) -> list[TableValueCandidate]:
        candidates: list[TableValueCandidate] = []

        for row in results:
            chunk = row.chunk
            for value_index, table_value in enumerate(chunk.table_values):
                if table_value.numeric_value is None:
                    continue

                candidates.append(
                    TableValueCandidate(
                        value_id=f"{chunk.chunk_id}:value:{value_index}",
                        chunk_id=chunk.chunk_id,
                        metric=table_value.metric,
                        table_column=table_value.table_column,
                        numeric_value=table_value.numeric_value,
                    )
                )

        return candidates

    def build_prompt(
        self,
        question: str,
        context: list[str],
        table_value_candidates: list[TableValueCandidate] | None = None,
    ) -> str:
        context_string = "\n\n---\n\n".join(context)
        candidate_payload = [
            candidate.model_dump() for candidate in table_value_candidates or []
        ]
        calculation_program_schema = json.dumps(
            CalculationProgram.model_json_schema(),
            indent=2,
        )
        return f"""
You are answering using only the provided retrieved context.

Rules:
- Output a single JSON object only.
- Do not wrap the output in markdown or extra prose.
- If the answer is not in the context, set answer to "I don't know".
- Do not invent numbers.
- Return exactly these keys: answer, citations, calculation_program.
- answer must be a scalar ConvFinQA-style answer only: a number, percentage, or short text value.
- Do not include units, explanatory prose, equations, markdown, or a full sentence in answer.
- citations must be a JSON array of chunk_id values used to support the answer.
- Set calculation_program to null only when the answer is a direct lookup or text answer.
- When arithmetic is needed, return a non-null calculation_program using only available_table_values or literal numbers from the question.
- For table values, set operand kind to "table_value" and value_id to one exact value_id from available_table_values.
- For prior step outputs, set operand kind to "step_result" and step_index to the zero-based prior step index.
- Be concise and precise.

Calculation program schema:
{calculation_program_schema}

Available table values for calculation:
{json.dumps(candidate_payload, indent=2)}

No-calculation response example:
{{
  "answer": "100",
  "citations": ["chunk_id_1"],
  "calculation_program": null
}}

Calculation response example:
{{
  "answer": "14.1%",
  "citations": ["chunk_id_1"],
  "calculation_program": {{
    "steps": [
      {{
        "operation": "subtract",
        "operands": [
          {{"kind": "table_value", "value_id": "chunk_id_1:value:0"}},
          {{"kind": "table_value", "value_id": "chunk_id_1:value:1"}}
        ]
      }},
      {{
        "operation": "divide",
        "operands": [
          {{"kind": "step_result", "step_index": 0}},
          {{"kind": "table_value", "value_id": "chunk_id_1:value:1"}}
        ]
      }}
    ]
  }}
}}

Question:
{question}

Retrieved context:
{context_string}
""".strip()

    def _parse_answer(self, output: str) -> RawRagAnswer:
        try:
            return RawRagAnswer.model_validate_json(output)
        except ValidationError:
            logger.warning("Model returned invalid RawRagAnswer JSON: %s", output)
            raise

    def build_final_answer(
        self,
        raw_answer: RawRagAnswer,
        table_value_candidates: list[TableValueCandidate],
    ) -> RagAnswer:
        final_answer = raw_answer.answer
        turn_programs = None

        if raw_answer.calculation_program is not None:
            calculation_trace = execute_calculation_program(
                raw_answer.calculation_program,
                table_value_candidates,
            )
            if calculation_trace.error is not None:
                raise ValueError(
                    f"Calculation program failed: {calculation_trace.error}"
                )
            if calculation_trace.turn_programs:
                turn_programs = ", ".join([*calculation_trace.turn_programs])

        return RagAnswer(
            answer=final_answer,
            citations=raw_answer.citations,
            turn_program=turn_programs,
        )

    def answer(self, question: str, record_id: str) -> RagAnswer:
        period_data = extract_period_data([question])
        results = self.retriever.retrieve(question, record_id, period_data)
        context_blocks = []

        for i, row in enumerate(results, start=1):
            chunk = row.chunk
            distance = row.distance
            context_blocks.append(self.build_context_block(i, chunk, distance))

        table_value_candidates = self.build_table_value_candidates(results)
        prompt = self.build_prompt(question, context_blocks, table_value_candidates)
        model_output = self.model_client.query_single(
            prompt,
            response_format=RawRagAnswer.model_json_schema(),
        )
        raw_answer = self._parse_answer(model_output.output)
        return self.build_final_answer(raw_answer, table_value_candidates)

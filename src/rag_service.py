import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.aggregator_service.executor import execute_calculation_program
from src.aggregator_service.query_intent import (
    CalculationProgram,
    TableValueCandidate,
)
from src.chunking_service.period_extraction import PeriodData, extract_period_data
from src.data_types import ChatHistoryPair, RetrievalChunk, RetrievedChunkRecord
from src.db_service.postgres_controllers import PostgresChunkStore
from src.logger import get_logger
from src.model_service.models import ModelClient, ModelOutput

logger = get_logger("rag_service")


def _format_calculated_answer(value: float, is_percentage: bool) -> str:
    if is_percentage:
        return f"{value:.1f}".rstrip("0").rstrip(".") + "%"
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


class RagAnswer(BaseModel):
    """Final scalar answer with supporting evidence and optional computation trace."""

    model_config = ConfigDict(extra="forbid")

    user_question: str | None = None
    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    calculation_program: CalculationProgram | None = Field(default=None, exclude=True)
    turn_program: str | None = None
    context_blocks: list[str] = Field(default_factory=list)
    tokens_used: int = Field(default=0)


class RAGService:
    def __init__(self, model_client: ModelClient, retriever: PostgresChunkStore):
        self.model_client = model_client
        self.retriever = retriever
        self.model_config = self.model_client.get_config()

    def parse_chat_history(
        self, chat_history: list[ChatHistoryPair] | None
    ) -> str | None:
        """Format previous turns and their evidence for prompt conditioning."""
        if not chat_history:
            return None
        parsed_history = []
        for turn_index, history in enumerate(reversed(chat_history), start=1):
            user_quistion = history.user_question
            parsed_history.append(f"User Qustion at Turn {turn_index + 1}: {user_quistion}")
        return "\n\n---\n\n".join(parsed_history)

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
        session_history: str | None = None,
    ) -> str:
        context_string = "\n\n---\n\n".join(context)
        session_history_string = session_history or "No prior turns."
        candidate_payload = [
            candidate.model_dump() for candidate in table_value_candidates or []
        ]
        calculation_program_schema = json.dumps(
            CalculationProgram.model_json_schema(),
            indent=2,
        )
        return f"""
You answer financial table questions using only the retrieved context.

Rules:
- Output a single JSON object only and Return exactly these keys: answer, citations, calculation_program.
- citations must contain the chunk_id that supports the answer.
- If the answer is not in the context, set answer to "I don't know" and do not invent numbers.
- Use prior turns only to resolve conversational follow-ups such as "what about in 2008?".
- Prefer the current retrieved context when it directly supports the answer.
- answer must be a scalar answer only: a number, percentage, or short text value.
- Do not include units, explanatory prose, equations, markdown, or a full sentence in answer.
- citations must be a JSON array of chunk_id values used to support the answer.
- Set calculation_program to null only when the answer is a direct lookup or text answer.
- When arithmetic is needed, return a non-null calculation_program using only available_table_values or literal numbers from the question.
- For table values, set operand kind to "table_value" and value_id to one exact value_id from available_table_values.
- For prior step outputs, set operand kind to "step_result" and step_index to the zero-based prior step index.
- For percent answers, include a final percentage step that converts the ratio to percent scale.

Calculation program schema:
{calculation_program_schema}

Available table values for calculation:
{json.dumps(candidate_payload, indent=2)}

Question:
{question}

Conversation history:
{session_history_string}

Retrieved context:
{context_string}

No-calculation response example:
{{
  "answer": "100",
  "citations": ["chunk_id_1"],
  "calculation_program": null
}}


Calculation response example:
{{
  "answer": "14.1",
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
      }},
      {{
        "operation": "percentage",
        "operands": [
          {{"kind": "step_result", "step_index": 1}}
        ]
      }}
    ]
  }}
}}

""".strip()

    def _parse_answer(self, model_output: ModelOutput) -> RagAnswer:
        try:
            answer = RagAnswer.model_validate_json(model_output.output)
        except ValidationError:
            logger.warning(
                "Model returned invalid RagAnswer JSON: %s",
                model_output.output,
            )
            raise
        return answer.model_copy(update={"tokens_used": model_output.tokens_used})

    def build_final_answer(
        self,
        model_answer: RagAnswer,
        table_value_candidates: list[TableValueCandidate],
        context_blocks: list[str],
        question: str,
    ) -> RagAnswer:
        final_answer = model_answer.answer
        turn_programs = None

        if (
            model_answer.calculation_program is not None
            and model_answer.calculation_program.steps
        ):
            calculation_trace = execute_calculation_program(
                model_answer.calculation_program,
                table_value_candidates,
            )
            if calculation_trace.error is not None:
                raise ValueError(
                    f"Calculation program failed: {calculation_trace.error}"
                )
            if calculation_trace.final_result is None:
                raise ValueError("Calculation program produced no final result")

            final_operation = model_answer.calculation_program.steps[-1].operation
            final_answer = _format_calculated_answer(
                calculation_trace.final_result,
                is_percentage=final_operation == "percentage",
            )
            if calculation_trace.turn_programs:
                turn_programs = ", ".join([*calculation_trace.turn_programs])

        return model_answer.model_copy(
            update={
                "answer": final_answer,
                "turn_program": turn_programs,
                "context_blocks": context_blocks,
            }
        )

    def answer(
        self,
        question: str,
        record_id: str,
        session_history: list[ChatHistoryPair] | None = None,
    ) -> RagAnswer:
        period_data = extract_period_data([question])
        results = self.retriever.retrieve(question, record_id, period_data)
        parsed_session_history = self.parse_chat_history(session_history)
        context_blocks = []

        for i, row in enumerate(results, start=1):
            chunk = row.chunk
            distance = row.distance
            context_blocks.append(self.build_context_block(i, chunk, distance))

        table_value_candidates = self.build_table_value_candidates(results)
        prompt = self.build_prompt(
            question,
            context_blocks,
            table_value_candidates,
            parsed_session_history,
        )
        logger.debug("RAG prompt:\n%s", prompt)
        model_output = self.model_client.query_single(
            prompt,
            response_format="json",
        )
        model_answer = self._parse_answer(model_output)
        return self.build_final_answer(
            model_answer, table_value_candidates, context_blocks, question
        )

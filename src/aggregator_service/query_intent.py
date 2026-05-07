from typing import Literal

from pydantic import BaseModel


class TableValueCandidate(BaseModel):
    """A numeric table value the model may reference in a calculation program."""

    value_id: str
    chunk_id: str
    metric: str
    table_column: str
    numeric_value: float


class Operand(BaseModel):
    kind: Literal["table_value", "literal", "step_result"]
    value_id: str | None = None
    literal: float | None = None
    step_index: int | None = None


class CalculationStep(BaseModel):
    operation: Literal[
        "lookup",
        "add",
        "subtract",
        "multiply",
        "divide",
        "sum",
        "average",
        "median",
        "percentage",
        "percentage_change",
    ]
    operands: list[Operand]


class CalculationProgram(BaseModel):
    steps: list[CalculationStep]


class CalculationStepTrace(BaseModel):
    step_index: int
    operation: str
    operands: list[float]
    result: float


class CalculationTrace(BaseModel):
    steps: list[CalculationStepTrace]
    turn_programs: list[str]
    final_result: float | None = None
    error: str | None = None

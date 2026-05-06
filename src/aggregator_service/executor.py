from src.aggregator_service.aggregators import OPERATIONS
from src.aggregator_service.query_intent import (
    CalculationProgram,
    CalculationStepTrace,
    CalculationTrace,
    Operand,
    TableValueCandidate,
)


def _resolve_operand(
    operand: Operand,
    value_lookup: dict[str, TableValueCandidate],
    step_results: list[float],
) -> float:
    if operand.kind == "table_value":
        if operand.value_id is None:
            raise ValueError("table_value operand is missing value_id")
        if operand.value_id not in value_lookup:
            raise ValueError(f"Unknown table value_id: {operand.value_id}")
        return value_lookup[operand.value_id].numeric_value

    if operand.kind == "literal":
        if operand.literal is None:
            raise ValueError("literal operand is missing literal")
        return operand.literal

    if operand.kind == "step_result":
        if operand.step_index is None:
            raise ValueError("step_result operand is missing step_index")
        if operand.step_index < 0 or operand.step_index >= len(step_results):
            raise ValueError(f"Unknown step result index: {operand.step_index}")
        return step_results[operand.step_index]

    raise ValueError(f"Unsupported operand kind: {operand.kind}")


def execute_calculation_program(
    program: CalculationProgram,
    candidates: list[TableValueCandidate],
) -> CalculationTrace:
    value_lookup = {candidate.value_id: candidate for candidate in candidates}
    step_results: list[float] = []
    step_traces: list[CalculationStepTrace] = []

    try:
        # TODO: This also assumes chainging -> we need an upstream change to the type
        for step_index, step in enumerate(program.steps):
            operation = OPERATIONS.get(step.operation)
            if operation is None:
                raise ValueError(f"Unsupported operation: {step.operation}")

            operands = [
                _resolve_operand(operand, value_lookup, step_results)
                for operand in step.operands
            ]
            result = operation(operands)
            step_results.append(result)
            step_traces.append(
                CalculationStepTrace(
                    step_index=step_index,
                    operation=step.operation,
                    operands=operands,
                    result=result,
                )
            )
    except Exception as exc:
        return CalculationTrace(
            steps=step_traces,
            final_result=step_results[-1] if step_results else None,
            error=str(exc),
        )

    return CalculationTrace(
        steps=step_traces,
        final_result=step_results[-1] if step_results else None,
    )

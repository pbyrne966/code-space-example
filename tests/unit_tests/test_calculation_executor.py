import unittest

from src.aggregator_service.executor import execute_calculation_program
from src.aggregator_service.query_intent import (
    CalculationProgram,
    CalculationStep,
    Operand,
    TableValueCandidate,
)


class CalculationExecutorTest(unittest.TestCase):
    def test_executes_multiple_steps_with_step_result_operand(self) -> None:
        candidates = [
            TableValueCandidate(
                value_id="chunk-1:value:0",
                chunk_id="chunk-1",
                metric="cash",
                table_column="2009",
                numeric_value=206588.0,
            ),
            TableValueCandidate(
                value_id="chunk-1:value:1",
                chunk_id="chunk-1",
                metric="cash",
                table_column="2008",
                numeric_value=181001.0,
            ),
        ]
        program = CalculationProgram(
            steps=[
                CalculationStep(
                    operation="subtract",
                    operands=[
                        Operand(kind="table_value", value_id="chunk-1:value:0"),
                        Operand(kind="table_value", value_id="chunk-1:value:1"),
                    ],
                ),
                CalculationStep(
                    operation="divide",
                    operands=[
                        Operand(kind="step_result", step_index=0),
                        Operand(kind="table_value", value_id="chunk-1:value:1"),
                    ],
                ),
            ]
        )

        trace = execute_calculation_program(program, candidates)

        self.assertIsNone(trace.error)
        self.assertEqual(trace.steps[0].result, 25587.0)
        self.assertAlmostEqual(trace.final_result or 0, 0.14136, places=5)

    def test_returns_error_trace_for_unknown_value_id(self) -> None:
        program = CalculationProgram(
            steps=[
                CalculationStep(
                    operation="lookup",
                    operands=[
                        Operand(kind="table_value", value_id="missing"),
                    ],
                )
            ]
        )

        trace = execute_calculation_program(program, [])

        self.assertEqual(trace.steps, [])
        self.assertIsNone(trace.final_result)
        self.assertEqual(trace.error, "Unknown table value_id: missing")


if __name__ == "__main__":
    unittest.main()

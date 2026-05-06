from statistics import median as statistics_median


def lookup(values: list[float]) -> float:
    if len(values) != 1:
        raise ValueError("lookup expects exactly one operand")
    return values[0]


def percentage(values: list[float]) -> float:
    if len(values) != 1:
        raise ValueError("percentage expects exactly one operand")
    return values[0] * 100


def percentage_change(values: list[float]) -> float:
    if len(values) != 2:
        raise ValueError("percentage_change expects exactly two operands")
    old_value, new_value = values
    if old_value == 0:
        raise ZeroDivisionError("Cannot calculate percentage change from zero")
    return (new_value - old_value) / old_value


def sum_values(values: list[float]) -> float:
    if not values:
        raise ValueError("sum expects at least one operand")
    return sum(values)


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("average expects at least one operand")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median expects at least one operand")
    return float(statistics_median(values))


def add(values: list[float]) -> float:
    return sum_values(values)


def subtract(values: list[float]) -> float:
    if len(values) != 2:
        raise ValueError("subtract expects exactly two operands")
    return values[0] - values[1]


def multiply(values: list[float]) -> float:
    if not values:
        raise ValueError("multiply expects at least one operand")
    result = 1.0
    for value in values:
        result *= value
    return result


def divide(values: list[float]) -> float:
    if len(values) != 2:
        raise ValueError("divide expects exactly two operands")
    if values[1] == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return values[0] / values[1]

OPERATIONS = {
    "lookup": lookup,
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
    "sum": sum_values,
    "average": average,
    "median": median,
    "percentage": percentage,
    "percentage_change": percentage_change,
}

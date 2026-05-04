import re
from dataclasses import dataclass, field
from datetime import date
from typing import TypeVar

T = TypeVar("T")

YEAR_MATCH = re.compile(r"\b\d{4}\b")
MONTH_NAME_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|sept(?:ember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
MONTH_MATCH = re.compile(rf"\b({MONTH_NAME_PATTERN})\.?\b", re.IGNORECASE)
MONTH_DAY_YEAR_MATCH = re.compile(
    rf"\b(?P<month>{MONTH_NAME_PATTERN})\.?\s+"
    r"(?P<day>[0-3]?\d)(?:st|nd|rd|th)?[,]?\s+"
    r"(?P<year>\d{4})\b",
    re.IGNORECASE,
)
DAY_MONTH_YEAR_MATCH = re.compile(
    r"\b(?P<day>[0-3]?\d)(?:st|nd|rd|th)?\s+"
    rf"(?P<month>{MONTH_NAME_PATTERN})\.?[,]?\s+"
    r"(?P<year>\d{4})\b",
    re.IGNORECASE,
)
QUARTER_MATCH = re.compile(
    r"\b(?:q([1-4])|quarter\s+([1-4])|([1-4])(?:st|nd|rd|th)\s+quarter)\b",
    re.IGNORECASE,
)
MIN_YEAR = 1900
MAX_YEAR = 2027
MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class PeriodData:
    years: list[str] = field(default_factory=list)
    months: list[int] = field(default_factory=list)
    quarters: list[int] = field(default_factory=list)
    days: list[int] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    period_labels: list[str] = field(default_factory=list)


def _dedupe_preserving_order[T](values: list[T]) -> list[T]:
    return list(dict.fromkeys(values))


def _extend_unique[T](target: list[T], values: list[T]) -> None:
    target[:] = _dedupe_preserving_order(target + values)


def _extract_years(label: str) -> list[str]:
    years: list[str] = []
    for raw_year in YEAR_MATCH.findall(label):
        year = int(raw_year)
        if MIN_YEAR <= year <= MAX_YEAR:
            years.append(raw_year)
    return _dedupe_preserving_order(years)


def _normalize_month_name(month_name: str) -> str:
    return month_name.lower().rstrip(".")


def _extract_months(label: str) -> list[int]:
    months: list[int] = []
    for match in MONTH_MATCH.finditer(label):
        month = MONTH_NAME_TO_NUMBER.get(_normalize_month_name(match.group(1)))
        if month is None:
            continue

        months.append(month)
    return _dedupe_preserving_order(months)


def _extract_quarters(label: str) -> list[int]:
    quarters: list[int] = []
    for match in QUARTER_MATCH.finditer(label):
        quarter_text = next(group for group in match.groups() if group is not None)
        if not quarter_text or not isinstance(quarter_text, str):
            continue

        quarters.append(int(quarter_text))
    return _dedupe_preserving_order(quarters)


def _extract_days_and_dates(label: str) -> tuple[list[int], list[str]]:
    days: list[int] = []
    dates: list[str] = []

    for date_pattern in (MONTH_DAY_YEAR_MATCH, DAY_MONTH_YEAR_MATCH):
        for match in date_pattern.finditer(label):
            day = int(match.group("day"))
            month = MONTH_NAME_TO_NUMBER[_normalize_month_name(match.group("month"))]
            year = int(match.group("year"))
            if not MIN_YEAR <= year <= MAX_YEAR:
                continue

            try:
                extracted_date = date(year, month, day)
            except ValueError:
                continue

            days.append(day)
            dates.append(extracted_date.isoformat())

    return _dedupe_preserving_order(days), _dedupe_preserving_order(dates)


def extract_period_data(labels: list[str]) -> PeriodData:
    period_data = PeriodData()

    for label in labels:
        label_years = _extract_years(label)
        label_months = _extract_months(label)
        label_quarters = _extract_quarters(label)
        label_days, label_dates = _extract_days_and_dates(label)

        if label_years or label_months or label_quarters or label_days or label_dates:
            _extend_unique(period_data.period_labels, [label])

        _extend_unique(period_data.years, label_years)
        _extend_unique(period_data.months, label_months)
        _extend_unique(period_data.quarters, label_quarters)
        _extend_unique(period_data.days, label_days)
        _extend_unique(period_data.dates, label_dates)

    return period_data

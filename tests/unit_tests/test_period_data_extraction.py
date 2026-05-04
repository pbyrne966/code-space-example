import unittest

from src.chunking_service.period_extraction import extract_period_data


class PeriodDataExtractionTest(unittest.TestCase):
    def test_extracts_month_day_date_and_year_metadata(self) -> None:
        period_data = extract_period_data(
            [
                "three months ended March 31, 2020",
                "year ended 31 December 2021",
                "Q4 2021",
            ]
        )

        self.assertEqual(period_data.years, ["2020", "2021"])
        self.assertEqual(period_data.months, [3, 12])
        self.assertEqual(period_data.quarters, [4])
        self.assertEqual(period_data.days, [31])
        self.assertEqual(period_data.dates, ["2020-03-31", "2021-12-31"])
        self.assertEqual(
            period_data.period_labels,
            [
                "three months ended March 31, 2020",
                "year ended 31 December 2021",
                "Q4 2021",
            ],
        )

    def test_skips_invalid_calendar_dates(self) -> None:
        period_data = extract_period_data(["month ended February 31, 2020"])

        self.assertEqual(period_data.years, ["2020"])
        self.assertEqual(period_data.months, [2])
        self.assertEqual(period_data.days, [])
        self.assertEqual(period_data.dates, [])


if __name__ == "__main__":
    unittest.main()

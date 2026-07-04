"""Design-win corpus schema validation and survival-table conversion."""

import pandas as pd
import pytest

from survivalsemi.designwins import load_designwins, to_survival_table


def write_corpus(tmp_path, rows, columns=None):
    columns = columns or ["ticker", "announce_date", "customer", "segment",
                          "source_url", "revenue_event_date"]
    path = tmp_path / "wins.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


GOOD_ROWS = [
    ["MCHP", "2024-03-01", "OEM A", "auto", "https://x", "2025-01-15"],
    ["NXPI", "2024-06-10", "OEM B", "industrial", "https://y", None],
]


class TestLoad:
    def test_good_corpus_loads(self, tmp_path):
        corpus = load_designwins(write_corpus(tmp_path, GOOD_ROWS))
        assert len(corpus) == 2
        assert corpus["announce_date"].dtype.kind == "M"

    def test_missing_column_rejected(self, tmp_path):
        path = write_corpus(
            tmp_path,
            [["MCHP", "2024-03-01", "auto", "https://x"]],
            columns=["ticker", "announce_date", "segment", "source_url"],
        )
        with pytest.raises(ValueError, match="missing required"):
            load_designwins(path)

    def test_unknown_segment_rejected(self, tmp_path):
        rows = [["MCHP", "2024-03-01", "OEM", "aerospace", "https://x", None]]
        with pytest.raises(ValueError, match="segments"):
            load_designwins(write_corpus(tmp_path, rows))

    def test_revenue_before_announcement_rejected(self, tmp_path):
        rows = [["MCHP", "2024-03-01", "OEM", "auto", "https://x", "2023-01-01"]]
        with pytest.raises(ValueError, match="precedes"):
            load_designwins(write_corpus(tmp_path, rows))


class TestSurvivalTable:
    def test_observed_and_censored(self, tmp_path):
        corpus = load_designwins(write_corpus(tmp_path, GOOD_ROWS))
        table = to_survival_table(corpus, as_of="2026-07-01")

        observed = table[table.event]
        censored = table[~table.event]
        assert len(observed) == 1 and len(censored) == 1
        # MCHP: 2024-03-01 -> 2025-01-15 = 320 days
        assert observed.iloc[0].duration_days == 320
        # NXPI censored at as_of: 2024-06-10 -> 2026-07-01 = 751 days
        assert censored.iloc[0].duration_days == 751

    def test_as_of_before_announcement_rejected(self, tmp_path):
        corpus = load_designwins(write_corpus(tmp_path, GOOD_ROWS))
        with pytest.raises(ValueError, match="as_of"):
            to_survival_table(corpus, as_of="2024-01-01")

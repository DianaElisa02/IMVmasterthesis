import pandas as pd
import pytest

from src.exposure_dimensions import _collapse_benefit_to_household


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "idhh": ["1"] * len(values),
            "dwt": [100.0] * len(values),
            "benefit": values,
        }
    )


def test_unique_positive_recovers_nonfirst_value() -> None:
    result = _collapse_benefit_to_household(
        _frame([0.0, 500.0, 0.0]),
        "benefit",
        "test",
        aggregation="unique_positive",
    )

    assert result.loc[0, "benefit"] == pytest.approx(500.0)


def test_unique_positive_rejects_conflicting_values() -> None:
    with pytest.raises(ValueError, match="multiple distinct positive"):
        _collapse_benefit_to_household(
            _frame([400.0, 500.0, 0.0]),
            "benefit",
            "test",
            aggregation="unique_positive",
        )


def test_sum_unique_positive_sums_distinct_values() -> None:
    result = _collapse_benefit_to_household(
        _frame([0.0, 522.96, 650.14, 0.0]),
        "benefit",
        "test",
        aggregation="sum_unique_positive",
    )

    assert result.loc[0, "benefit"] == pytest.approx(1173.10)


def test_sum_unique_positive_deduplicates_identical_values() -> None:
    result = _collapse_benefit_to_household(
        _frame([500.0, 500.0, 0.0]),
        "benefit",
        "test",
        aggregation="sum_unique_positive",
    )

    assert result.loc[0, "benefit"] == pytest.approx(500.0)


def test_components_can_be_combined_after_separate_aggregation() -> None:
    df = pd.DataFrame(
        {
            "idhh": ["1", "1", "1"],
            "dwt": [100.0, 100.0, 100.0],
            "bsa00_s": [0.0, 450.0, 0.0],
            "bsarg_s": [300.0, 0.0, 0.0],
        }
    )

    imv = _collapse_benefit_to_household(
        df,
        "bsa00_s",
        "IMV",
        aggregation="unique_positive",
    ).rename(columns={"bsa00_s": "bsa00_hh"})

    regional = _collapse_benefit_to_household(
        df,
        "bsarg_s",
        "regional",
        aggregation="sum_unique_positive",
    )[["idhh", "bsarg_s"]].rename(columns={"bsarg_s": "bsarg_hh"})

    combined = imv.merge(regional, on="idhh", validate="one_to_one")
    combined["total_post"] = combined["bsa00_hh"] + combined["bsarg_hh"]

    assert combined.loc[0, "total_post"] == pytest.approx(750.0)

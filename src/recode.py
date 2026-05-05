"""
recode.py
=========
All recoding logic for the ECV → EUROMOD UDB conversion pipeline.

Each function takes one or more polars Series and returns a Series.
No file I/O is performed here. All mappings reference constants.py.
"""

from __future__ import annotations

import polars as pl

from src.constants import (
    DDI_DISABLED,
    DDI_NOT_APPLICABLE,
    DDI_NOT_DISABLED,
    DEH_RECODE_BOUNDARIES,
    DEH_DEFAULT,
    DGN_DEFAULT,
    DGN_VALID_VALUES,
    DRGN1_MAP,
    DRGN2_MAP,
    DMS_DEFAULT,
    DMS_RECODE,
    LES_DEFAULT,
    LINDI_DEFAULT,
    LINDI_MAP,
    PL031_TO_LES,
    PL040_TO_LES,
)


def _const(value: float, n: int) -> pl.Series:
    """Return a Float64 Series of length n filled with value."""
    return pl.Series([value] * n, dtype=pl.Float64)


def recode_drgn1(db040: pl.Series) -> pl.Series:
    return (
        db040.cast(pl.String)
        .replace(
            list(DRGN1_MAP.keys()),
            [str(v) for v in DRGN1_MAP.values()],
            default=None,
        )
        .cast(pl.Float64)
    )


def recode_drgn2(db040: pl.Series) -> pl.Series:
    return (
        db040.cast(pl.String)
        .replace(
            list(DRGN2_MAP.keys()),
            [str(v) for v in DRGN2_MAP.values()],
            default=None,
        )
        .cast(pl.Float64)
    )


def recode_dgn(rb090: pl.Series) -> pl.Series:
    num = rb090.cast(pl.Float64, strict=False)
    is_valid = num.is_in(list(DGN_VALID_VALUES)).fill_null(False)
    return num.zip_with(is_valid, _const(float(DGN_DEFAULT), len(num)))


def recode_dms(pb190: pl.Series, idpartner: pl.Series) -> pl.Series:  # noqa: ARG001
    # idpartner not needed: all remaining nulls fall back to DMS_DEFAULT anyway
    num = pb190.cast(pl.Int64, strict=False)
    return (
        num.replace(list(DMS_RECODE.keys()), list(DMS_RECODE.values()), default=None)
        .cast(pl.Float64)
        .fill_null(float(DMS_DEFAULT))
    )


def recode_deh(pe040: pl.Series) -> pl.Series:
    num = pe040.cast(pl.Float64, strict=False)
    result = _const(float(DEH_DEFAULT), len(num))
    for low, high, target in DEH_RECODE_BOUNDARIES:
        mask = ((num >= low) & (num <= high)).fill_null(False)
        result = _const(float(target), len(num)).zip_with(mask, result)
    return result


def recode_ddi(pl031: pl.Series, has_personal_record: pl.Series) -> pl.Series:
    """
    has_personal_record: boolean Series, True for persons with a TP record (adults).
    False/null indicates children / no personal info collected.
    """
    pl031_num = pl031.cast(pl.Float64, strict=False)

    is_disabled     = (pl031_num == 8.0).fill_null(False)
    is_not_disabled = pl031_num.is_not_null() & ~(pl031_num == 8.0)
    is_not_applicable = pl031_num.is_null() & ~has_personal_record.fill_null(False)

    n = len(pl031)
    result = pl.Series([None] * n, dtype=pl.Float64)
    result = _const(float(DDI_DISABLED),     n).zip_with(is_disabled,           result)
    result = _const(float(DDI_NOT_DISABLED), n).zip_with(is_not_disabled,       result)
    result = _const(float(DDI_NOT_APPLICABLE), n).zip_with(is_not_applicable,   result)
    return result


def compute_dag(rb080: pl.Series, rb010: pl.Series) -> pl.Series:
    birth_year  = rb080.cast(pl.Float64, strict=False)
    survey_year = rb010.cast(pl.Float64, strict=False)
    return (survey_year - birth_year - 1).clip(lower_bound=0).cast(pl.Float64)


def compute_oecd_m(person_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute OECD modified equivalence scale per household.

    Expects columns IDHH (String) and dag (Float64).
    Returns a DataFrame with IDHH and oecd_m.
    """
    age = person_df["dag"].cast(pl.Float64, strict=False)
    temp = pl.DataFrame({
        "IDHH":   person_df["IDHH"],
        "_adult": (age >= 14).fill_null(False).cast(pl.Int32),
        "_child": (age < 14).fill_null(False).cast(pl.Int32),
    })
    agg = temp.group_by("IDHH").agg([
        pl.col("_adult").sum(),
        pl.col("_child").sum(),
    ])
    return agg.with_columns(
        (1.0 + (pl.col("_adult") - 1).clip(lower_bound=0) * 0.5 + pl.col("_child") * 0.3)
        .alias("oecd_m")
    ).select(["IDHH", "oecd_m"])


def recode_les(pl031: pl.Series, pl040: pl.Series, dag: pl.Series) -> pl.Series:
    pl031_num = pl031.cast(pl.Int64, strict=False)
    pl040_num = pl040.cast(pl.Int64, strict=False)
    age       = dag.cast(pl.Float64, strict=False)

    n = len(pl031)
    result = pl.Series([None] * n, dtype=pl.Int64)

    child_mask = (age < 6).fill_null(False)
    result = pl.Series([0] * n, dtype=pl.Int64).zip_with(child_mask, result)

    for val, les_val in PL031_TO_LES.items():
        mask = (pl031_num == val).fill_null(False) & result.is_null()
        result = pl.Series([les_val] * n, dtype=pl.Int64).zip_with(mask, result)

    for val, les_val in PL040_TO_LES.items():
        mask = (pl040_num == val).fill_null(False) & result.is_null()
        result = pl.Series([les_val] * n, dtype=pl.Int64).zip_with(mask, result)

    return result.fill_null(LES_DEFAULT).cast(pl.Float64)


def recode_lindi(pl111a: pl.Series) -> pl.Series:
    cleaned = pl111a.cast(pl.String).str.strip_chars().str.to_lowercase()
    return (
        cleaned.replace(
            list(LINDI_MAP.keys()),
            [str(v) for v in LINDI_MAP.values()],
            default=None,
        )
        .cast(pl.Float64)
        .fill_null(float(LINDI_DEFAULT))
    )


def recode_amrtn(hh021: pl.Series) -> pl.Series:
    num = hh021.cast(pl.Int64, strict=False)
    result = num.clone()
    n = len(num)
    result = pl.Series([2] * n, dtype=pl.Int64).zip_with((num == 1).fill_null(False), result)
    result = pl.Series([1] * n, dtype=pl.Int64).zip_with((num == 2).fill_null(False), result)
    result = pl.Series([6] * n, dtype=pl.Int64).zip_with((num == 5).fill_null(False), result)
    return result.cast(pl.Float64)


def scale_monthly(annual: pl.Series, hx010: pl.Series) -> pl.Series:
    ann   = annual.cast(pl.Float64, strict=False).fill_null(0.0)
    scale = hx010.cast(pl.Float64, strict=False).fill_null(1.0)
    return (ann / 12.0 * scale).cast(pl.Float64)


def scale_monthly_hh(annual: pl.Series, hx010: pl.Series) -> pl.Series:
    ann   = annual.cast(pl.Float64, strict=False).fill_null(0.0)
    scale = hx010.cast(pl.Float64, strict=False).fill_null(1.0)
    return (ann / 12.0 * scale).cast(pl.Float64)


def compute_liwmy(
    pl073: pl.Series, pl074: pl.Series, pl075: pl.Series, pl076: pl.Series,
) -> pl.Series:
    tmp = pl.DataFrame({
        "a": pl073.cast(pl.Float64, strict=False),
        "b": pl074.cast(pl.Float64, strict=False),
        "c": pl075.cast(pl.Float64, strict=False),
        "d": pl076.cast(pl.Float64, strict=False),
    })
    total = tmp.select(pl.sum_horizontal("a", "b", "c", "d")).to_series()
    return total.clip(upper_bound=12).fill_null(0.0).cast(pl.Float64)


def compute_liwftmy(pl073: pl.Series, pl075: pl.Series) -> pl.Series:
    tmp = pl.DataFrame({
        "a": pl073.cast(pl.Float64, strict=False),
        "b": pl075.cast(pl.Float64, strict=False),
    })
    total = tmp.select(pl.sum_horizontal("a", "b")).to_series()
    return total.clip(upper_bound=12).fill_null(0.0).cast(pl.Float64)


def compute_liwptmy(pl074: pl.Series, pl076: pl.Series) -> pl.Series:
    tmp = pl.DataFrame({
        "a": pl074.cast(pl.Float64, strict=False),
        "b": pl076.cast(pl.Float64, strict=False),
    })
    total = tmp.select(pl.sum_horizontal("a", "b")).to_series()
    return total.clip(upper_bound=12).fill_null(0.0).cast(pl.Float64)


def compute_liwwh(pl200: pl.Series) -> pl.Series:
    years = pl200.cast(pl.Float64, strict=False)
    return (years * 12).clip(lower_bound=0, upper_bound=780).fill_null(0.0).cast(pl.Float64)


def fill_zero(series: pl.Series) -> pl.Series:
    return series.cast(pl.Float64, strict=False).fill_null(0.0)

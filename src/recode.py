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
    DEH_DEFAULT,
    DEH_RECODE_BOUNDARIES,
    DGN_DEFAULT,
    DGN_VALID_VALUES,
    DMS_DEFAULT,
    DMS_RECODE,
    DRGN1_MAP,
    DRGN2_MAP,
    LES_DEFAULT,
    LINDI_DEFAULT,
    LINDI_MAP,
    PL031_TO_LES,
    PL040_TO_LES,
)


def _const(value: float, n: int) -> pl.Series:
    """Return a Float64 Series of length n filled with value."""
    return pl.Series([value] * n, dtype=pl.Float64)


def recode_drgn1(db040: pl.Expr) -> pl.Expr:
    return (
        db040.cast(pl.String)
        .replace(
            list(DRGN1_MAP.keys()),
            [str(v) for v in DRGN1_MAP.values()],
            default=None,
        )
        .cast(pl.Float64)
    )


def recode_drgn2(db040: pl.Expr) -> pl.Expr:
    return (
        db040.cast(pl.String)
        .replace(
            list(DRGN2_MAP.keys()),
            [str(v) for v in DRGN2_MAP.values()],
            default=None,
        )
        .cast(pl.Float64)
    )


def recode_dgn(rb090: pl.Expr) -> pl.Expr:
    num = rb090.cast(pl.Float64, strict=False)
    is_valid = num.is_in(list(DGN_VALID_VALUES)).fill_null(False)

    return (
        pl.when(is_valid)
        .then(num)
        .otherwise(pl.lit(float(DGN_DEFAULT), dtype=pl.Float64))
        .cast(pl.Float64)
    )


def recode_dms(pb190: pl.Expr, idpartner: pl.Expr) -> pl.Expr:
    # idpartner not needed: all remaining nulls fall back to DMS_DEFAULT anyway
    num = pb190.cast(pl.Int64, strict=False)

    return (
        num.replace(
            list(DMS_RECODE.keys()),
            list(DMS_RECODE.values()),
            default=None,
        )
        .cast(pl.Float64)
        .fill_null(float(DMS_DEFAULT))
    )


def recode_deh(pe040: pl.Expr) -> pl.Expr:
    num = pe040.cast(pl.Float64, strict=False)

    result = pl.lit(float(DEH_DEFAULT), dtype=pl.Float64)

    for low, high, target in DEH_RECODE_BOUNDARIES:
        mask = ((num >= low) & (num <= high)).fill_null(False)

        result = (
            pl.when(mask)
            .then(pl.lit(float(target), dtype=pl.Float64))
            .otherwise(result)
        )

    return result.cast(pl.Float64)


def recode_ddi(
    pl031: pl.Expr,
    has_personal_record: pl.Expr,
) -> pl.Expr:
    """
    has_personal_record: boolean expression, True for persons with a TP record.
    False/null indicates children / no personal info collected.
    """
    pl031_num = pl031.cast(pl.Float64, strict=False)
    has_record = has_personal_record.cast(pl.Boolean).fill_null(False)

    is_disabled = (pl031_num == 8.0).fill_null(False)

    is_not_disabled = pl031_num.is_not_null() & (pl031_num != 8.0).fill_null(False)

    is_not_applicable = pl031_num.is_null() & ~has_record

    return (
        pl.when(is_disabled)
        .then(pl.lit(float(DDI_DISABLED), dtype=pl.Float64))
        .when(is_not_disabled)
        .then(pl.lit(float(DDI_NOT_DISABLED), dtype=pl.Float64))
        .when(is_not_applicable)
        .then(pl.lit(float(DDI_NOT_APPLICABLE), dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .cast(pl.Float64)
    )


def compute_dag(birth_year: pl.Expr, survey_year: pl.Expr) -> pl.Expr:
    return (survey_year - birth_year - 1).clip(lower_bound=0).cast(pl.Float64)


def compute_oecd_m(person_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute OECD modified equivalence scale per household.

    Expects columns IDHH (String) and dag (Float64).
    Returns a DataFrame with IDHH and oecd_m.
    """
    age = person_df["dag"].cast(pl.Float64, strict=False)
    temp = pl.DataFrame(
        {
            "IDHH": person_df["IDHH"],
            "_adult": (age >= 14).fill_null(False).cast(pl.Int32),
            "_child": (age < 14).fill_null(False).cast(pl.Int32),
        }
    )
    agg = temp.group_by("IDHH").agg(
        [
            pl.col("_adult").sum(),
            pl.col("_child").sum(),
        ]
    )
    return agg.with_columns(
        (
            1.0
            + (pl.col("_adult") - 1).clip(lower_bound=0) * 0.5
            + pl.col("_child") * 0.3
        ).alias("oecd_m")
    ).select(["IDHH", "oecd_m"])


def recode_les(
    pl031: pl.Expr,
    pl040: pl.Expr,
    dag: pl.Expr,
) -> pl.Expr:
    pl031_num = pl031.cast(pl.Int64, strict=False)
    pl040_num = pl040.cast(pl.Int64, strict=False)
    age = dag.cast(pl.Float64, strict=False)

    from_pl031 = pl031_num.replace(
        list(PL031_TO_LES.keys()),
        list(PL031_TO_LES.values()),
        default=None,
    ).cast(pl.Int64)

    from_pl040 = pl040_num.replace(
        list(PL040_TO_LES.keys()),
        list(PL040_TO_LES.values()),
        default=None,
    ).cast(pl.Int64)

    return (
        pl.when((age < 6).fill_null(False))
        .then(pl.lit(0, dtype=pl.Int64))
        .otherwise(
            pl.coalesce(
                from_pl031,
                from_pl040,
                pl.lit(LES_DEFAULT, dtype=pl.Int64),
            )
        )
        .cast(pl.Float64)
    )


def recode_lindi(pl111a: pl.Expr) -> pl.Expr:
    cleaned = pl111a.cast(pl.String, strict=False).str.strip_chars().str.to_lowercase()

    return (
        cleaned.replace(
            list(LINDI_MAP.keys()),
            [str(v) for v in LINDI_MAP.values()],
            default=None,
        )
        .cast(pl.Float64)
        .fill_null(float(LINDI_DEFAULT))
    )


def recode_amrtn(hh021: pl.Expr) -> pl.Expr:
    num = hh021.cast(pl.Int64, strict=False)

    return num.replace(
        {
            1: 2,
            2: 1,
            5: 6,
        }
    ).cast(pl.Float64)


def scale_monthly(annual: pl.Expr, hx010: pl.Expr) -> pl.Expr:
    ann = annual.cast(pl.Float64, strict=False).fill_null(0.0)
    scale = hx010.cast(pl.Float64, strict=False).fill_null(1.0)

    return (ann / 12.0 * scale).cast(pl.Float64)


def compute_liwmy(
    pl073: pl.Expr,
    pl074: pl.Expr,
    pl075: pl.Expr,
    pl076: pl.Expr,
) -> pl.Expr:
    total = pl.sum_horizontal(
        pl073.cast(pl.Float64, strict=False),
        pl074.cast(pl.Float64, strict=False),
        pl075.cast(pl.Float64, strict=False),
        pl076.cast(pl.Float64, strict=False),
    )

    return total.clip(upper_bound=12).fill_null(0.0).cast(pl.Float64)


def compute_liwftmy(
    pl073: pl.Expr,
    pl075: pl.Expr,
) -> pl.Expr:
    total = pl.sum_horizontal(
        pl073.cast(pl.Float64, strict=False),
        pl075.cast(pl.Float64, strict=False),
    )

    return total.clip(upper_bound=12).fill_null(0.0).cast(pl.Float64)


def compute_liwptmy(
    pl074: pl.Expr,
    pl076: pl.Expr,
) -> pl.Expr:
    total = pl.sum_horizontal(
        pl074.cast(pl.Float64, strict=False),
        pl076.cast(pl.Float64, strict=False),
    )

    return total.clip(upper_bound=12).fill_null(0.0).cast(pl.Float64)


def compute_liwwh(pl200: pl.Expr) -> pl.Expr:
    years = pl200.cast(pl.Float64, strict=False)

    return (
        (years * 12)
        .clip(
            lower_bound=0,
            upper_bound=780,
        )
        .fill_null(0.0)
        .cast(pl.Float64)
    )


def fill_zero(x: pl.Expr) -> pl.Expr:
    return x.cast(pl.Float64, strict=False).fill_null(0.0)

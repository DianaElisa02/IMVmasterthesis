"""
person.py
=========
Builds the person-level UDB variables from raw ECV Tr and Tp sections.

Returns a Polars DataFrame with one row per person. The OECD equivalence
scale is computed from age data and joined back before returning.
"""

from __future__ import annotations

import logging

import polars as pl

from src.recode import (
    compute_dag,
    compute_liwftmy,
    compute_liwmy,
    compute_liwptmy,
    compute_liwwh,
    compute_oecd_m,
    fill_zero,
    recode_ddi,
    recode_deh,
    recode_dgn,
    recode_dms,
    recode_les,
    recode_lindi,
    scale_monthly,
)

logger = logging.getLogger(__name__)


def recode_loc_expr(isco: pl.Expr) -> pl.Expr:
    """Map ISCO-08 occupation code to EUROMOD loc category."""
    num = isco.cast(pl.Int64, strict=False).fill_null(0)

    result = pl.lit(-1.0, dtype=pl.Float64)

    result = (
        pl.when(num.is_in([1, 2, 3]).fill_null(False))
        .then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(result)
    )

    for tens in range(1, 10):
        lo, hi = tens * 10, tens * 10 + 9
        mask = ((num >= lo) & (num <= hi)).fill_null(False)

        result = (
            pl.when(mask).then(pl.lit(float(tens), dtype=pl.Float64)).otherwise(result)
        )

    return result.cast(pl.Float64)


def recode_dcz_expr(pb220a: pl.Expr) -> pl.Expr:
    num = pb220a.cast(pl.Int64, strict=False)

    return (
        num.replace([1, 2, 3], [1, 2, 3], default=None).cast(pl.Float64).fill_null(1.0)
    )


def zero_to_null_expr(x: pl.Expr) -> pl.Expr:
    return (
        pl.when(x == 0.0)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(x)
        .cast(pl.Float64)
    )


def prepare_person_input(tr: pl.DataFrame, tp: pl.DataFrame, year: int) -> pl.DataFrame:
    person = tr.join(tp, left_on="RB030", right_on="PB030", how="left")

    n_tr = len(tr)
    n_merged = len(person)
    if n_merged != n_tr:
        logger.warning(
            "Year %s: row count changed after Tr-Tp join (%d → %d)",
            year,
            n_tr,
            n_merged,
        )

    return person.with_columns(
        pl.lit(0.0).alias("dwt"),  # DB090 Missing
        pl.lit(0.0).alias("lunwh"),  # PL271 missing
        pl.lit(0.0).alias("bunct"),
        pl.lit(0.0).alias("bunnc"),
        pl.lit(0.0).alias("bunot"),
        pl.lit(0.0).alias("bhl00"),
        pl.lit(0.0).alias("bhlot"),
        pl.lit(0.0).alias("pdi00"),
        pl.lit(0.0).alias("pdicm"),
        pl.lit(0.0).alias("pdinc"),
        pl.lit(0.0).alias("poa00"),
        pl.lit(0.0).alias("poacm"),
        pl.lit(0.0).alias("poanc"),
        pl.lit(0.0).alias("psuwd00"),
        pl.lit(0.0).alias("psuwdcm"),
        pl.lit(year).alias("year"),
        pl.lit(0.0).alias("pdiot"),
    )


def build_person_udb(person: pl.DataFrame, year: int) -> pl.DataFrame:
    out = (
        person.lazy()
        .with_columns(
            _rb010=pl.col("RB010").cast(pl.Float64, strict=False).fill_null(year),
            _rb030=pl.col("RB030").cast(pl.Int64, strict=False),
            _has_tp=pl.col("PB040").is_not_null(),
            _hx010=pl.lit(1.0, dtype=pl.Float64),
            _lhw_sum=(
                pl.col("PL060").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("PL100").cast(pl.Float64, strict=False).fill_null(0.0)
            ),
        )
        .with_columns(
            _dag=compute_dag(pl.col("RB080"), pl.col("_rb010")),
            _idpartner=fill_zero(pl.col("RB240")),
        )
        .select(
            (pl.col("RB030").cast(pl.Int64, strict=False) // 100)
            .cast(pl.String)
            .alias("IDHH"),
            pl.col("RB030")
            .cast(pl.Int64, strict=False)
            .cast(pl.String)
            .alias("idperson"),
            fill_zero(pl.col("RB230")).alias("idmother"),
            fill_zero(pl.col("RB220")).alias("idfather"),
            fill_zero(pl.col("RB240")).alias("idpartner"),
            fill_zero(pl.col("RB070")).alias("dmb"),
            (
                pl.col("PY010G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("yem"),
            (
                pl.col("PY050G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("yse"),
            (
                pl.col("PY080G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("ypp"),
            (
                pl.col("PY020G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("kfb"),
            (
                pl.col("PY021G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("kfbcc"),
            (
                pl.col("PY090G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("bun"),
            (
                pl.col("PY120G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("bhl"),
            (
                pl.col("PY130G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("pdi"),
            (
                pl.col("PY100G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("poa"),
            (
                pl.col("PY110G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("psu"),
            (
                pl.col("PY140G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("bed"),
            (
                pl.col("PY030G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("tscer"),
            (
                pl.col("PY035G").cast(pl.Float64, strict=False).fill_null(0.0) / 12.0
            ).alias("xpp"),
            pl.col("_dag").alias("dag"),
            recode_dgn(pl.col("RB090")).alias("dgn"),
            pl.lit(13.0, dtype=pl.Float64).alias("dct"),
            recode_ddi(pl.col("PL031"), pl.col("_has_tp")).alias("ddi"),
            recode_deh(pl.col("PE040")).alias("deh"),
            recode_dms(pl.col("PB190"), pl.col("_idpartner")).alias("dms"),
            recode_les(
                pl.col("PL031"),
                pl.col("PL040"),
                pl.col("_dag"),
            ).alias("les"),
            recode_dcz_expr(pl.col("PB220A")).alias("dcz"),
            zero_to_null_expr(pl.col("_lhw_sum"))
            .clip(lower_bound=1.0, upper_bound=80.0)
            .fill_null(0.0)
            .cast(pl.Float64)
            .alias("lhw"),
            compute_liwmy(
                pl.col("PL073"), pl.col("PL074"), pl.col("PL075"), pl.col("PL076")
            ).alias("liwmy"),
            compute_liwftmy(
                pl.col("PL073"),
                pl.col("PL075"),
            ).alias("liwftmy"),
            compute_liwptmy(
                pl.col("PL074"),
                pl.col("PL076"),
            ).alias("liwptmy"),
            compute_liwwh(pl.col("PL200")).alias("liwwh"),
            pl.col("PL080")
            .cast(pl.Float64, strict=False)
            .clip(upper_bound=12)
            .fill_null(0.0)
            .cast(pl.Float64)
            .alias("lunmy"),
            pl.col("PL085")
            .cast(pl.Float64, strict=False)
            .clip(upper_bound=12)
            .fill_null(0.0)
            .cast(pl.Float64)
            .alias("lpemy"),
            recode_lindi(pl.col("PL111A")).alias("lindi"),
            recode_loc_expr(pl.col("PL051")).alias("loc"),
            pl.col("PL040")
            .cast(pl.Int64, strict=False)
            .replace([1, 2, 3, 4], [1, 2, 0, 3], default=None)
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("lse"),
            pl.col("PL051")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .is_in([1, 2, 3, 11, 23, 34, 54])
            .cast(pl.Float64)
            .alias("lcs"),
        )
        .collect()
    )

    oecd_df = compute_oecd_m(out.select(["IDHH", "dag"]))
    out = out.join(oecd_df, on="IDHH", how="left").with_columns(
        pl.col("oecd_m").fill_null(1.0)
    )

    logger.info("Year %s: built person UDB — %d persons", year, len(out))
    return out

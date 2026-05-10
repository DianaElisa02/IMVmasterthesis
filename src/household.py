from __future__ import annotations

import logging

import polars as pl

from src.recode import (
    fill_zero,
    recode_amrtn,
    recode_drgn1,
    recode_drgn2,
    scale_monthly,
)

logger = logging.getLogger(__name__)

_ABSENT_FROM_ECV: frozenset[str] = frozenset({"DB070", "HH071", "HX090"})

_GROSS_INCOME_COLS: tuple[str, ...] = (
    "HY040G", "HY050G", "HY060G", "HY070G", "HY080G",
    "HY090G", "HY100G", "HY110G", "HY120G", "HY130G",
)


def prepare_household_input(
    td: pl.DataFrame, th: pl.DataFrame, year: int
) -> pl.DataFrame:
    for key, df, label in [("DB030", td, "Td"), ("HB030", th, "Th")]:
        n_unique = df.select(key).n_unique()
        if n_unique != len(df):
            raise ValueError(
                f"Year {year}: duplicate {key} keys in {label} "
                f"({len(df)} rows, {n_unique} unique)"
            )

    hh = td.join(th, left_on="DB030", right_on="HB030", how="inner")

    n_td = len(td)
    n_merged = len(hh)
    if n_merged < n_td:
        logger.warning(
            "Year %s: %d Td households did not match Th (%d Td, %d merged)",
            year, n_td - n_merged, n_td, n_merged,
        )

    for col in _ABSENT_FROM_ECV:
        if col in hh.columns:
            n = hh[col].is_not_null().sum()
            if n > 0:
                logger.warning(
                    "Year %s: %s expected absent from Spanish ECV but has %d non-null values",
                    year, col, n,
                )

    for col in _GROSS_INCOME_COLS:
        if col in hh.columns:
            n_neg = (hh[col].cast(pl.Float64, strict=False) < 0).fill_null(False).sum()
            if n_neg > 0:
                logger.warning(
                    "Year %s: %s has %d negative values (expected >= 0 for gross income)",
                    year, col, n_neg,
                )

    hh = hh.with_columns(
        pl.col("HX040").cast(pl.Float64, strict=False).fill_null(1.0).alias("_hsize_raw"),
    )

    n_bad = (hh["_hsize_raw"] <= 0).sum()
    if n_bad > 0:
        raise ValueError(
            f"Year {year}: {n_bad} households have HX040 <= 0"
        )

    return hh


def build_household_udb(hh: pl.DataFrame, year: int) -> pl.DataFrame:
    out = (
        hh.lazy()
        .select(
            pl.col("DB030").cast(pl.Int64, strict=False).cast(pl.String).alias("IDHH"),
            recode_drgn1(pl.col("DB040")).alias("drgn1"),
            recode_drgn2(pl.col("DB040")).alias("drgn2"),
            pl.col("DB090").alias("dwt"),
            pl.lit(13.0).alias("dct"),
            (pl.col("DB100").cast(pl.Float64, strict=False) == 2.0)
            .cast(pl.Float64)
            .fill_null(strategy="zero")
            .alias("drgmd"),
            (pl.col("DB100").cast(pl.Float64, strict=False) == 3.0)
            .cast(pl.Float64)
            .fill_null(strategy="zero")
            .alias("drgru"),
            (pl.col("DB100").cast(pl.Float64, strict=False) == 1.0)
            .cast(pl.Float64)
            .fill_null(strategy="zero")
            .alias("drgur"),
            fill_zero(pl.col("DB060")).alias("dsu01"),
            pl.col("HX040").cast(pl.Float64, strict=False).alias("hsize"),
            pl.when(
                pl.col("HH010").cast(pl.Float64, strict=False).is_null()
                | (pl.col("HH010").cast(pl.Float64, strict=False) == 0.0)
            )
            .then(pl.lit(1.0, dtype=pl.Float64))
            .otherwise(pl.col("HH010").cast(pl.Float64, strict=False))
            .alias("hh010"),
            fill_zero(pl.col("HH021")).alias("hh021"),
            fill_zero(pl.col("HH030")).alias("hh030"),
            fill_zero(pl.col("HH040")).alias("hh040"),
            fill_zero(pl.col("HH030")).alias("amrrm"),
            fill_zero(pl.col("HH050")).alias("amraw"),
            fill_zero(pl.col("HS021")).alias("amrub"),
            fill_zero(pl.col("HS090")).alias("aco"),
            fill_zero(pl.col("HS110")).alias("aca"),
            fill_zero(pl.col("HY020")).alias("hy020"),
            fill_zero(pl.col("HY022")).alias("hy022"),
            fill_zero(pl.col("HY023")).alias("hy023"),
            recode_amrtn(pl.col("HH021")).alias("amrtn"),
            scale_monthly(pl.col("HY020")).alias("yds"),
            scale_monthly(pl.col("HY090G")).alias("yiy"),
            scale_monthly(pl.col("HY040G")).alias("ypr"),
            scale_monthly(pl.col("HY080G")).alias("ypt"),
            scale_monthly(pl.col("HY050G")).alias("bfa"),
            scale_monthly(pl.col("HY070G")).alias("bho"),
            scale_monthly(pl.col("HY060G")).alias("bsa"),
            scale_monthly(pl.col("HY145N")).alias("tad"),
            scale_monthly(pl.col("HY120G")).alias("tpr"),
            scale_monthly(pl.col("HY120G")).alias("twl"),
            scale_monthly(pl.col("HY130G")).alias("xmp"),
            scale_monthly(pl.col("HY100G")).alias("xhcmomi"),
            scale_monthly(pl.col("HY110G")).alias("yot"),
            pl.col("HH060")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("xhcrt"),
            pl.col("HH070")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("xhc"),
            pl.lit(year, dtype=pl.Int32).alias("year"),
        )
        .with_columns(
            (pl.col("xhc") - pl.col("xhcrt") - pl.col("xhcmomi"))
            .clip(lower_bound=0.0)
            .alias("xhcot")
        )
        .collect()
    )

    n_fixed = hh.filter(
        pl.col("HH010").cast(pl.Float64, strict=False).is_null()
        | (pl.col("HH010").cast(pl.Float64, strict=False) == 0.0)
    ).height
    if n_fixed > 0:
        logger.info(
            "Year %s: recoded hh010=0/null to 1 for %d households",
            year, n_fixed,
        )

    logger.info("Year %s: built household UDB — %d households", year, len(out))
    return out
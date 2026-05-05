"""
household.py
============
Builds the household-level UDB variables from raw ECV Td and Th sections.

Returns a Polars DataFrame with one row per household. Merged onto the
person-level file in merge.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.readers import read_td, read_th
from src.recode import (
    fill_zero_expr,
    recode_amrtn_expr,
    recode_drgn1,
    recode_drgn2,
    scale_monthly_expr,
)

logger = logging.getLogger(__name__)


def build_household_udb(input_dir: Path, year: int) -> pl.DataFrame:
    td = read_td(input_dir, year)
    th = read_th(input_dir, year)

    hh = td.join(th, left_on="DB030", right_on="HB030", how="inner")

    n_td = len(td)
    n_merged = len(hh)
    if n_merged < n_td:
        logger.warning(
            "Year %s: %d Td households did not match Th (%d Td, %d merged)",
            year,
            n_td - n_merged,
            n_td,
            n_merged,
        )

    c = Cols(set(hh.columns))
    db100 = c.f64("DB100")

    hh = hh.with_columns(
        pl.lit(1.0, dtype=pl.Float64).alias("HX010"),
        pl.lit(0.0, dtype=pl.Float64).alias("bma"),
        pl.lit(0.0, dtype=pl.Float64).alias("bch"),
        pl.lit(0.0, dtype=pl.Float64).alias("bch00"),
        pl.lit(0.0, dtype=pl.Float64).alias("bchdi"),
        pl.lit(0.0, dtype=pl.Float64).alias("bchot"),
        pl.lit(0.0, dtype=pl.Float64).alias("xpp"),
        pl.lit(0.0).alias("dsu00"),
        pl.lit(0.0).alias("xhcmomc"),
        pl.lit(0.0).alias("yptmp"),
        pl.lit(0.0).alias("tis"),
        pl.lit(0.0).alias("afc"),
        pl.lit(0.0).alias("tintrch"),
    )

    out = (
        hh.lazy()
        .select(
            pl.col("DB030").cast(pl.Int64, strict=False).cast(pl.String).alias("IDHH"),
            recode_drgn1(pl.col("DB040")).alias("drgn1"),
            recode_drgn2(pl.col("DB040")).alias("drgn2"),
            pl.col("DB090").alias("dwt"),
            pl.lit(13.0).alias("dct"),  # Introduce dct column fixed at 13
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
            fill_zero_expr(pl.col("DB060")).alias("dsu01"),
            pl.col("HX040").cast(pl.Float64, strict=False).alias("hsize"),
            fill_zero_expr(pl.col("HH010")).alias("hh010"),
            fill_zero_expr(pl.col("HH021")).alias("hh021"),
            fill_zero_expr(pl.col("HH030")).alias("hh030"),
            fill_zero_expr(pl.col("HH040")).alias("hh040"),
            fill_zero_expr(pl.col("HH030")).alias("amrrm"),
            fill_zero_expr(pl.col("HH050")).alias("amraw"),
            fill_zero_expr(pl.col("HS021")).alias("amrub"),
            fill_zero_expr(pl.col("HS090")).alias("aco"),
            fill_zero_expr(pl.col("HS110")).alias("aca"),
            fill_zero_expr(pl.col("HY020")).alias("hy020"),
            fill_zero_expr(pl.col("HY022")).alias("hy022"),
            fill_zero_expr(pl.col("HY023")).alias("hy023"),
            recode_amrtn_expr(pl.col("HH021")).alias("amrtn"),
            scale_monthly_expr(pl.col("HY020"), pl.col("HX010")).alias("yds"),
            scale_monthly_expr(pl.col("HY090G"), pl.col("HX010")).alias("yiy"),
            scale_monthly_expr(pl.col("HY040G"), pl.col("HX010")).alias("ypr"),
            scale_monthly_expr(pl.col("HY080G"), pl.col("HX010")).alias("ypt"),
            scale_monthly_expr(pl.col("HY050G"), pl.col("HX010")).alias("bfa"),
            scale_monthly_expr(pl.col("HY070G"), pl.col("HX010")).alias("bho"),
            scale_monthly_expr(pl.col("HY060G"), pl.col("HX010")).alias("bsa"),
            scale_monthly_expr(pl.col("HY145N"), pl.col("HX010")).alias("tad"),
            scale_monthly_expr(pl.col("HY120G"), pl.col("HX010")).alias("tpr"),
            scale_monthly_expr(pl.col("HY120G"), pl.col("HX010")).alias("twl"),
            scale_monthly_expr(pl.col("HY130G"), pl.col("HX010")).alias("xmp"),
            scale_monthly_expr(pl.col("HY100G"), pl.col("HX010")).alias("xhcmomi"),
            (pl.col("HH060").cast(pl.Float64, strict=False) * pl.col("HX010"))
            .fill_null(0.0)
            .alias("xhcrt"),
            (pl.col("HH070").cast(pl.Float64, strict=False) * pl.col("HX010"))
            .fill_null(0.0)
            .alias("xhc"),
            scale_monthly_expr(pl.col("HY110G"), pl.col("HX010")).alias("yot"),
            pl.lit(year, dtype=pl.Int32).alias("year"),
        )
        .with_columns(
            (pl.col("xhc") - pl.col("xhcrt") - pl.col("xhcmomi"))
            .clip(lower_bound=0.0)
            .alias("xhcot")
        )
        .collect()
    )

    logger.info("Year %s: built household UDB — %d households", year, len(out))
    return out

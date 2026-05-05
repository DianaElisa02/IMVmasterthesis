"""
household.py
============
Builds the household-level UDB variables from raw ECV Td and Th sections.

Returns a Polars DataFrame with one row per household. Merged onto the
person-level file in merge.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.readers import read_td, read_th
from src.recode import (
    fill_zero,
    recode_amrtn,
    recode_drgn1,
    recode_drgn2,
    scale_monthly_hh,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cols:
    available: set[str]

    def has(self, name: str) -> bool:
        return name in self.available

    def f64(self, name: str, default: float = 0.0) -> pl.Expr:
        if self.has(name):
            return pl.col(name).cast(pl.Float64, strict=False)
        return pl.lit(default, dtype=pl.Float64)

    def string(self, name: str, default: str = "") -> pl.Expr:
        if self.has(name):
            return pl.col(name).cast(pl.String, strict=False)
        return pl.lit(default, dtype=pl.String)

    def const_f64(self, value: float) -> pl.Expr:
        return pl.lit(value, dtype=pl.Float64)


def _get(df: pl.DataFrame, col: str) -> pl.Series:
    """Return column cast to Float64, or a zero-filled series if absent."""
    if col in df.columns:
        return df[col].cast(pl.Float64, strict=False)
    return pl.Series(col, [0.0] * len(df), dtype=pl.Float64)


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

    out = (
        hh.lazy()
        .select(
            pl.col("DB030").cast(pl.Int64, strict=False).cast(pl.String).alias("IDHH"),
            recode_drgn1(c.string("DB040")).alias("drgn1"),
            recode_drgn2(c.string("DB040")).alias("drgn2"),
            c.f64("DB090").alias("dwt"),
            c.const_f64(13.0).alias("dct"),
            (db100 == 2.0).cast(pl.Float64).fill_null(0.0).alias("drgmd"),
            (db100 == 3.0).cast(pl.Float64).fill_null(0.0).alias("drgru"),
            (db100 == 1.0).cast(pl.Float64).fill_null(0.0).alias("drgur"),
        )
        .collect()
    )

    cols: dict[str, pl.Series] = {name: out[name] for name in out.columns}

    cols["dsu00"] = fill_zero(_get(hh, "DB070"))
    cols["dsu01"] = fill_zero(_get(hh, "DB060"))

    cols["hsize"] = _get(hh, "HX040")
    cols["hh010"] = fill_zero(_get(hh, "HH010"))
    cols["hh021"] = fill_zero(_get(hh, "HH021"))
    cols["hh030"] = fill_zero(_get(hh, "HH030"))
    cols["hh040"] = fill_zero(_get(hh, "HH040"))

    cols["amrtn"] = (
        recode_amrtn(_get(hh, "HH021"))
        if "HH021" in hh.columns
        else pl.Series([0.0] * len(hh), dtype=pl.Float64)
    )
    cols["amrrm"] = fill_zero(_get(hh, "HH030"))
    cols["amraw"] = fill_zero(_get(hh, "HH050"))
    cols["amrub"] = fill_zero(_get(hh, "HS021"))
    cols["aco"] = fill_zero(_get(hh, "HS090"))
    cols["aca"] = fill_zero(_get(hh, "HS110"))

    cols["hy020"] = fill_zero(_get(hh, "HY020"))
    cols["hy022"] = fill_zero(_get(hh, "HY022"))
    cols["hy023"] = fill_zero(_get(hh, "HY023"))

    hx010 = _get(hh, "HX010").fill_null(1.0)
    cols["yds"] = scale_monthly_hh(_get(hh, "HY020"), hx010)
    cols["yiy"] = scale_monthly_hh(_get(hh, "HY090G"), hx010)
    cols["ypr"] = scale_monthly_hh(_get(hh, "HY040G"), hx010)
    cols["ypt"] = scale_monthly_hh(_get(hh, "HY080G"), hx010)
    cols["yptmp"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)
    cols["bfa"] = scale_monthly_hh(_get(hh, "HY050G"), hx010)
    cols["bho"] = scale_monthly_hh(_get(hh, "HY070G"), hx010)
    cols["bsa"] = scale_monthly_hh(_get(hh, "HY060G"), hx010)
    cols["bma"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)
    cols["bch"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)
    cols["bch00"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)
    cols["bchdi"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)
    cols["bchot"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)

    cols["tad"] = scale_monthly_hh(_get(hh, "HY145N"), hx010)
    cols["tis"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)
    cols["tpr"] = scale_monthly_hh(_get(hh, "HY120G"), hx010)
    cols["twl"] = scale_monthly_hh(_get(hh, "HY120G"), hx010)
    cols["xmp"] = scale_monthly_hh(_get(hh, "HY130G"), hx010)
    cols["xpp"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)

    hy100g = _get(hh, "HY100G")
    hh060 = _get(hh, "HH060")
    hh070 = _get(hh, "HH070")

    cols["xhcmomi"] = scale_monthly_hh(hy100g, hx010)
    cols["xhcmomc"] = fill_zero(_get(hh, "HH071"))
    cols["xhcrt"] = (hh060 * hx010).fill_null(0.0)
    cols["xhc"] = (hh070 * hx010).fill_null(0.0)
    cols["xhcot"] = (cols["xhc"] - cols["xhcrt"] - cols["xhcmomi"]).clip(
        lower_bound=0.0
    )

    cols["yot"] = scale_monthly_hh(_get(hh, "HY110G"), hx010)
    cols["afc"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)
    cols["tintrch"] = pl.Series([0.0] * len(hh), dtype=pl.Float64)

    cols["year"] = pl.Series([year] * len(hh), dtype=pl.Int32)

    out = pl.DataFrame(cols)
    logger.info("Year %s: built household UDB — %d households", year, len(out))
    return out

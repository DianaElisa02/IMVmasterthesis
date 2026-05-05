"""
person.py
=========
Builds the person-level UDB variables from raw ECV Tr and Tp sections.

Returns a Polars DataFrame with one row per person. The OECD equivalence
scale is computed from age data and joined back before returning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.readers import read_tr, read_tp
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
    recode_les,
    recode_lindi,
    recode_dms,
    scale_monthly,
)

logger = logging.getLogger(__name__)


def _get(df: pl.DataFrame, col: str) -> pl.Series:
    """Return column cast to Float64, or a zero-filled series if absent."""
    if col in df.columns:
        return df[col].cast(pl.Float64, strict=False)
    return pl.Series(col, [0.0] * len(df), dtype=pl.Float64)


def _get_raw(df: pl.DataFrame, col: str) -> pl.Series:
    """Return column as-is, or a null String series if absent."""
    if col in df.columns:
        return df[col]
    return pl.Series(col, [None] * len(df), dtype=pl.String)


def _recode_loc(isco: pl.Series) -> pl.Series:
    """Map ISCO-08 occupation code to EUROMOD loc category (0–9, -1=unknown)."""
    num = isco.cast(pl.Int64, strict=False).fill_null(0)
    n   = len(num)
    result = pl.Series([-1.0] * n, dtype=pl.Float64)
    result = pl.Series([0.0] * n, dtype=pl.Float64).zip_with(
        num.is_in([1, 2, 3]).fill_null(False), result
    )
    for tens in range(1, 10):
        lo, hi = tens * 10, tens * 10 + 9
        mask = ((num >= lo) & (num <= hi)).fill_null(False)
        result = pl.Series([float(tens)] * n, dtype=pl.Float64).zip_with(mask, result)
    return result


def _recode_dcz(pb220a: pl.Series) -> pl.Series:
    num = pb220a.cast(pl.Int64, strict=False)
    return (
        num.replace([1, 2, 3], [1, 2, 3], default=None)
        .cast(pl.Float64)
        .fill_null(1.0)
    )


def _zero_to_null(s: pl.Series) -> pl.Series:
    """Replace exact 0.0 with null; used for lhw computation."""
    return (
        pl.DataFrame({"s": s})
        .select(
            pl.when(pl.col("s") == 0.0).then(None).otherwise(pl.col("s")).alias("r")
        )
        .to_series()
        .cast(pl.Float64)
    )


def build_person_udb(input_dir: Path, year: int) -> pl.DataFrame:
    tr = read_tr(input_dir, year)
    tp = read_tp(input_dir, year)

    person = tr.join(tp, left_on="RB030", right_on="PB030", how="left")

    n_tr     = len(tr)
    n_merged = len(person)
    if n_merged != n_tr:
        logger.warning(
            "Year %s: row count changed after Tr-Tp join (%d → %d)",
            year, n_tr, n_merged,
        )

    rb010 = (
        _get_raw(person, "RB010").cast(pl.String).fill_null(str(year))
        .cast(pl.Float64, strict=False)
    )
    rb030 = person["RB030"].cast(pl.Int64, strict=False)

    # True for persons with a TP record (adults with personal income data)
    has_tp = (
        person["PB040"].is_not_null() if "PB040" in person.columns
        else pl.Series([False] * len(person), dtype=pl.Boolean)
    )

    hx010 = pl.Series([1.0] * len(person), dtype=pl.Float64)

    cols: dict[str, pl.Series] = {}

    cols["IDHH"]      = (rb030 // 100).cast(pl.String)
    cols["idperson"]  = rb030.cast(pl.String)
    cols["idmother"]  = fill_zero(_get_raw(person, "RB230"))
    cols["idfather"]  = fill_zero(_get_raw(person, "RB220"))
    cols["idpartner"] = fill_zero(_get_raw(person, "RB240"))

    dag = compute_dag(_get(person, "RB080"), rb010)
    cols["dag"] = dag
    cols["dgn"] = recode_dgn(_get(person, "RB090"))
    cols["dct"] = pl.Series([13.0] * len(person), dtype=pl.Float64)
    cols["dmb"] = fill_zero(_get_raw(person, "RB070"))
    cols["dwt"] = fill_zero(_get_raw(person, "DB090"))

    cols["ddi"] = recode_ddi(_get_raw(person, "PL031"), has_tp)
    cols["deh"] = recode_deh(_get(person, "PE040"))
    cols["dms"] = recode_dms(_get_raw(person, "PB190"), cols["idpartner"])

    cols["les"] = recode_les(
        _get_raw(person, "PL031"),
        _get_raw(person, "PL040"),
        dag,
    )

    pl060    = _get(person, "PL060").fill_null(0.0)
    pl100    = _get(person, "PL100").fill_null(0.0)
    lhw_sum  = pl060 + pl100
    cols["lhw"] = (
        _zero_to_null(lhw_sum)
        .clip(lower_bound=1.0, upper_bound=80.0)
        .fill_null(0.0)
        .cast(pl.Float64)
    )

    cols["liwmy"]   = compute_liwmy(
        _get_raw(person, "PL073"), _get_raw(person, "PL074"),
        _get_raw(person, "PL075"), _get_raw(person, "PL076"),
    )
    cols["liwftmy"] = compute_liwftmy(_get_raw(person, "PL073"), _get_raw(person, "PL075"))
    cols["liwptmy"] = compute_liwptmy(_get_raw(person, "PL074"), _get_raw(person, "PL076"))
    cols["liwwh"]   = compute_liwwh(_get_raw(person, "PL200"))
    cols["lunmy"]   = _get(person, "PL080").clip(upper_bound=12).fill_null(0.0).cast(pl.Float64)
    cols["lunwh"]   = fill_zero(_get_raw(person, "PL271"))
    cols["lpemy"]   = _get(person, "PL085").clip(upper_bound=12).fill_null(0.0).cast(pl.Float64)

    cols["lindi"] = recode_lindi(_get_raw(person, "PL111A"))

    cols["loc"] = _recode_loc(_get(person, "PL051"))

    cols["lse"] = (
        _get_raw(person, "PL040")
        .cast(pl.Int64, strict=False)
        .replace([1, 2, 3, 4], [1, 2, 0, 3], default=None)
        .cast(pl.Float64)
        .fill_null(0.0)
    )

    civil_servant_codes = [1, 2, 3, 11, 23, 34, 54]
    cols["lcs"] = (
        _get(person, "PL051").cast(pl.Int64, strict=False).fill_null(0)
        .is_in(civil_servant_codes)
        .cast(pl.Float64)
    )

    cols["yem"]   = scale_monthly(_get(person, "PY010G"), hx010)
    cols["yse"]   = scale_monthly(_get(person, "PY050G"), hx010)
    cols["ypp"]   = scale_monthly(_get(person, "PY080G"), hx010)
    cols["kfb"]   = scale_monthly(_get(person, "PY020G"), hx010)
    cols["kfbcc"] = scale_monthly(_get(person, "PY021G"), hx010)

    cols["bun"]   = scale_monthly(_get(person, "PY090G"), hx010)
    cols["bunct"] = pl.Series([0.0] * len(person), dtype=pl.Float64)
    cols["bunnc"] = pl.Series([0.0] * len(person), dtype=pl.Float64)
    cols["bunot"] = pl.Series([0.0] * len(person), dtype=pl.Float64)

    cols["bhl"]   = scale_monthly(_get(person, "PY120G"), hx010)
    cols["bhl00"] = pl.Series([0.0] * len(person), dtype=pl.Float64)
    cols["bhlot"] = pl.Series([0.0] * len(person), dtype=pl.Float64)

    cols["pdi"]   = scale_monthly(_get(person, "PY130G"), hx010)
    cols["pdi00"] = pl.Series([0.0] * len(person), dtype=pl.Float64)
    cols["pdicm"] = pl.Series([0.0] * len(person), dtype=pl.Float64)
    cols["pdinc"] = pl.Series([0.0] * len(person), dtype=pl.Float64)
    cols["pdiot"] = pl.Series([0.0] * len(person), dtype=pl.Float64)

    cols["poa"]   = scale_monthly(_get(person, "PY100G"), hx010)
    cols["poa00"] = pl.Series([0.0] * len(person), dtype=pl.Float64)
    cols["poacm"] = pl.Series([0.0] * len(person), dtype=pl.Float64)
    cols["poanc"] = pl.Series([0.0] * len(person), dtype=pl.Float64)

    cols["psu"]     = scale_monthly(_get(person, "PY110G"), hx010)
    cols["psuwd00"] = pl.Series([0.0] * len(person), dtype=pl.Float64)
    cols["psuwdcm"] = pl.Series([0.0] * len(person), dtype=pl.Float64)

    cols["bed"]   = scale_monthly(_get(person, "PY140G"), hx010)
    cols["tscer"] = scale_monthly(_get(person, "PY030G"), hx010)
    cols["xpp"]   = scale_monthly(_get(person, "PY035G"), hx010)

    cols["dcz"] = _recode_dcz(_get_raw(person, "PB220A"))

    cols["year"] = pl.Series([year] * len(person), dtype=pl.Int32)

    interim = pl.DataFrame(cols)
    oecd_df = compute_oecd_m(interim.select(["IDHH", "dag"]))
    out = interim.join(oecd_df, on="IDHH", how="left").with_columns(
        pl.col("oecd_m").fill_null(1.0)
    )

    logger.info("Year %s: built person UDB — %d persons", year, len(out))
    return out

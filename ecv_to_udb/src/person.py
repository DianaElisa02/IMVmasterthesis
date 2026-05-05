"""
person.py
=========
Builds the person-level UDB variables from raw ECV Tr and Tp sections.

Returns a DataFrame with one row per person containing all person-level
UDB variables. The OECD equivalence scale is computed here from age data
and merged back onto persons before returning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

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


def _get(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(0.0, index=df.index, dtype="float64")


def _get_raw(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def build_person_udb(input_dir: Path, year: int) -> pd.DataFrame:
    tr = read_tr(input_dir, year)
    tp = read_tp(input_dir, year)

    person = tr.merge(tp, left_on="RB030", right_on="PB030", how="left", suffixes=("_tr", "_tp"))

    n_tr = len(tr)
    n_merged = len(person)
    if n_merged != n_tr:
        logger.warning(
            "Year %s: row count changed after Tr-Tp merge (%d → %d)",
            year, n_tr, n_merged,
        )

    rb010 = _get_raw(person, "RB010").fillna(str(year))

    out = pd.DataFrame(index=person.index)

    rb030 = pd.to_numeric(person["RB030"], errors="coerce")
    out["IDHH"]     = rb030.floordiv(100).astype("Int64").astype("string")
    out["idperson"] = rb030.astype("Int64").astype("string")
    out["idmother"]  = fill_zero(_get_raw(person, "RB230"))
    out["idfather"]  = fill_zero(_get_raw(person, "RB220"))
    out["idpartner"] = fill_zero(_get_raw(person, "RB240"))

    out["dag"] = compute_dag(_get(person, "RB080"), pd.to_numeric(rb010, errors="coerce"))
    out["dgn"] = recode_dgn(_get(person, "RB090"))
    out["dct"] = 13.0
    out["dmb"] = fill_zero(_get_raw(person, "RB070"))
    out["dwt"] = fill_zero(_get_raw(person, "DB090"))

    out["ddi"] = recode_ddi(
    _get_raw(person, "PL031"),
    _get_raw(person, "PB030"),
    )
    out["deh"] = recode_deh(_get(person, "PE040"))
    out["dms"] = recode_dms(
        _get_raw(person, "PB190"),
        out["idpartner"],
    )

    out["les"] = recode_les(
    _get_raw(person, "PL031"),
    _get_raw(person, "PL040"),
    out["dag"],
    )

    out["lhw"] = (
        _get(person, "PL060").add(_get(person, "PL100"), fill_value=0.0)
        .replace(0.0, np.nan)
        .clip(lower=1.0, upper=80.0)
        .fillna(0.0)
        .astype("Float64")
    )

    out["liwmy"]   = compute_liwmy(
        _get_raw(person, "PL073"), _get_raw(person, "PL074"),
        _get_raw(person, "PL075"), _get_raw(person, "PL076"),
    )
    out["liwftmy"] = compute_liwftmy(
        _get_raw(person, "PL073"), _get_raw(person, "PL075"),
    )
    out["liwptmy"] = compute_liwptmy(
        _get_raw(person, "PL074"), _get_raw(person, "PL076"),
    )
    out["liwwh"]   = compute_liwwh(_get_raw(person, "PL200"))
    out["lunmy"]   = _get(person, "PL080").clip(upper=12).fillna(0.0).astype("Float64")
    out["lunwh"]   = fill_zero(_get_raw(person, "PL271"))
    out["lpemy"]   = _get(person, "PL085").clip(upper=12).fillna(0.0).astype("Float64")

    out["lindi"] = recode_lindi(_get_raw(person, "PL111A"))

    pl051 = pd.to_numeric(_get_raw(person, "PL051"), errors="coerce").fillna(0).astype(int)
    out["loc"] = pl051.apply(_recode_loc).astype("Float64")

    pl040 = pd.to_numeric(_get_raw(person, "PL040"), errors="coerce")
    out["lse"] = pl040.map({1: 1.0, 2: 2.0, 3: 0.0, 4: 3.0}).fillna(0.0).astype("Float64")

    civil_servant_codes = {1, 2, 3, 11, 23, 34, 54}
    out["lcs"] = pl051.isin(civil_servant_codes).astype(float)

    hx010 = pd.Series(1.0, index=person.index)

    out["yem"]   = scale_monthly(_get(person, "PY010G"), hx010)
    out["yse"]   = scale_monthly(_get(person, "PY050G"), hx010)
    out["ypp"]   = scale_monthly(_get(person, "PY080G"), hx010)
    out["kfb"]   = scale_monthly(_get(person, "PY020G"), hx010)
    out["kfbcc"] = scale_monthly(_get(person, "PY021G"), hx010)

    out["bun"]  = scale_monthly(_get(person, "PY090G"), hx010)
    out["bunct"] = 0.0
    out["bunnc"] = 0.0
    out["bunot"] = 0.0

    out["bhl"]   = scale_monthly(_get(person, "PY120G"), hx010)
    out["bhl00"] = 0.0
    out["bhlot"] = 0.0

    out["pdi"]   = scale_monthly(_get(person, "PY130G"), hx010)
    out["pdi00"] = 0.0
    out["pdicm"] = 0.0
    out["pdinc"] = 0.0
    out["pdiot"] = 0.0

    out["poa"]   = scale_monthly(_get(person, "PY100G"), hx010)
    out["poa00"] = 0.0
    out["poacm"] = 0.0
    out["poanc"] = 0.0

    out["psu"]     = scale_monthly(_get(person, "PY110G"), hx010)
    out["psuwd00"] = 0.0
    out["psuwdcm"] = 0.0

    out["bed"] = scale_monthly(_get(person, "PY140G"), hx010)
    out["tscer"] = scale_monthly(_get(person, "PY030G"), hx010)
    out["xpp"]   = scale_monthly(_get(person, "PY035G"), hx010)

    out["dcz"] = _recode_dcz(_get_raw(person, "PB220A"))

    oecd = compute_oecd_m(out[["IDHH", "DAG"]].rename(columns={"IDHH": "IDHH", "DAG": "DAG"}) if "DAG" in out.columns else out.assign(DAG=out["dag"])[["IDHH", "DAG"]])
    out = out.merge(
        oecd.reset_index().rename(columns={"IDHH": "IDHH", "oecd_m": "oecd_m"}),
        on="IDHH",
        how="left",
    )
    out["oecd_m"] = out["oecd_m"].fillna(1.0).astype("Float64")

    out["year"] = year

    logger.info("Year %s: built person UDB — %d persons", year, len(out))
    return out


def _recode_loc(isco: int) -> float:
    if isco in (1, 2, 3):
        return 0.0
    if 10 <= isco <= 19:
        return 1.0
    if 20 <= isco <= 29:
        return 2.0
    if 30 <= isco <= 39:
        return 3.0
    if 40 <= isco <= 49:
        return 4.0
    if 50 <= isco <= 59:
        return 5.0
    if 60 <= isco <= 69:
        return 6.0
    if 70 <= isco <= 79:
        return 7.0
    if 80 <= isco <= 89:
        return 8.0
    if 90 <= isco <= 99:
        return 9.0
    return -1.0


def _recode_dcz(pb220a: pd.Series) -> pd.Series:
    num = pd.to_numeric(pb220a, errors="coerce").astype("Int64")
    result = num.map({1: 1, 2: 2, 3: 3})
    return result.fillna(1).astype("Float64")
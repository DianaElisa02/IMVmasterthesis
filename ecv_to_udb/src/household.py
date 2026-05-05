"""
household.py
============
Builds the household-level UDB variables from raw ECV Td and Th sections.

Returns a DataFrame with one row per household containing all household-level
UDB variables. This is later merged onto the person-level file in merge.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.readers import read_td, read_th
from src.recode import (
    fill_zero,
    recode_amrtn,
    recode_drgn1,
    recode_drgn2,
    scale_monthly_hh,
)

logger = logging.getLogger(__name__)


def _get(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(0.0, index=df.index, dtype="float64")


def build_household_udb(input_dir: Path, year: int) -> pd.DataFrame:
    td = read_td(input_dir, year)
    th = read_th(input_dir, year)

    hh = td.merge(th, left_on="DB030", right_on="HB030", how="inner")

    n_td = len(td)
    n_merged = len(hh)
    if n_merged < n_td:
        logger.warning(
            "Year %s: %d Td households did not match Th (%d Td, %d merged)",
            year, n_td - n_merged, n_td, n_merged,
        )

    hx010 = _get(hh, "HX010").fillna(1.0)

    out = pd.DataFrame(index=hh.index)

    out["IDHH"] = pd.to_numeric(hh["DB030"], errors="coerce").astype("Int64").astype("string")

    out["drgn1"] = recode_drgn1(hh["DB040"])
    out["drgn2"] = recode_drgn2(hh["DB040"])
    out["dwt"]   = _get(hh, "DB090")
    out["dct"]   = 13.0

    out["drgmd"] = np.where(_get(hh, "DB100").eq(2), 1.0, 0.0)
    out["drgru"] = np.where(_get(hh, "DB100").eq(3), 1.0, 0.0)
    out["drgur"] = np.where(_get(hh, "DB100").eq(1), 1.0, 0.0)

    out["dsu00"] = fill_zero(hh.get("DB070", pd.Series(0, index=hh.index)))
    out["dsu01"] = fill_zero(hh.get("DB060", pd.Series(0, index=hh.index)))

    out["hsize"]  = _get(hh, "HX040")
    out["hh010"]  = fill_zero(hh.get("HH010", pd.Series(np.nan, index=hh.index)))
    out["hh021"]  = fill_zero(hh.get("HH021", pd.Series(np.nan, index=hh.index)))
    out["hh030"]  = fill_zero(hh.get("HH030", pd.Series(np.nan, index=hh.index)))
    out["hh040"]  = fill_zero(hh.get("HH040", pd.Series(np.nan, index=hh.index)))
    out["amrtn"]  = recode_amrtn(hh["HH021"]) if "HH021" in hh.columns else pd.Series(0.0, index=hh.index)
    out["amrrm"]  = fill_zero(hh.get("HH030", pd.Series(np.nan, index=hh.index)))
    out["amraw"]  = fill_zero(hh.get("HH050", pd.Series(np.nan, index=hh.index)))
    out["amrub"]  = fill_zero(hh.get("HS021", pd.Series(np.nan, index=hh.index)))
    out["aco"]    = fill_zero(hh.get("HS090", pd.Series(np.nan, index=hh.index)))
    out["aca"]    = fill_zero(hh.get("HS110", pd.Series(np.nan, index=hh.index)))

    out["hy020"]  = fill_zero(hh.get("HY020", pd.Series(np.nan, index=hh.index)))
    out["hy022"]  = fill_zero(hh.get("HY022", pd.Series(np.nan, index=hh.index)))
    out["hy023"]  = fill_zero(hh.get("HY023", pd.Series(np.nan, index=hh.index)))

    out["yds"]    = scale_monthly_hh(_get(hh, "HY020"), hx010)
    out["yiy"]    = scale_monthly_hh(_get(hh, "HY090G"), hx010)
    out["ypr"]    = scale_monthly_hh(_get(hh, "HY040G"), hx010)
    out["ypt"]    = scale_monthly_hh(_get(hh, "HY080G"), hx010)
    out["yptmp"]  = 0.0
    out["bfa"]    = scale_monthly_hh(_get(hh, "HY050G"), hx010)
    out["bho"]    = scale_monthly_hh(_get(hh, "HY070G"), hx010)
    out["bsa"]    = scale_monthly_hh(_get(hh, "HY060G"), hx010)
    out["bma"]    = 0.0
    out["bch"]    = 0.0
    out["bch00"]  = 0.0
    out["bchdi"]  = 0.0
    out["bchot"]  = 0.0

    out["tad"]    = scale_monthly_hh(_get(hh, "HY145N"), hx010)
    out["tis"]    = 0.0
    out["tpr"]    = scale_monthly_hh(_get(hh, "HY120G"), hx010)
    out["twl"]    = scale_monthly_hh(_get(hh, "HY120G"), hx010)
    out["xmp"]    = scale_monthly_hh(_get(hh, "HY130G"), hx010)
    out["xpp"]    = 0.0

    hy100g = _get(hh, "HY100G")
    hh060  = _get(hh, "HH060")
    hh070  = _get(hh, "HH070")

    out["xhcmomi"] = scale_monthly_hh(hy100g, hx010)
    out["xhcmomc"] = fill_zero(hh.get("HH071", pd.Series(np.nan, index=hh.index)))
    out["xhcrt"]   = (hh060 * hx010).fillna(0.0)
    out["xhc"]     = (hh070 * hx010).fillna(0.0)
    out["xhcot"]   = (out["xhc"] - out["xhcrt"] - out["xhcmomi"]).clip(lower=0.0)

    out["yot"]     = scale_monthly_hh(_get(hh, "HY110G"), hx010)

    out["afc"]     = 0.0
    out["tintrch"] = 0.0

    out["year"] = year

    logger.info("Year %s: built household UDB — %d households", year, len(out))
    return out
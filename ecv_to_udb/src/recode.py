"""
recode.py
=========
All recoding logic for the ECV → EUROMOD UDB conversion pipeline.

Each function takes one or more pandas Series and returns a Series.
No file I/O is performed here. All mappings reference constants.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def recode_drgn1(db040: pd.Series) -> pd.Series:
    return db040.astype("string").map(DRGN1_MAP).astype("Float64")


def recode_drgn2(db040: pd.Series) -> pd.Series:
    return db040.astype("string").map(DRGN2_MAP).astype("Float64")


def recode_dgn(rb090: pd.Series) -> pd.Series:
    num = pd.to_numeric(rb090, errors="coerce")
    valid = num.isin(DGN_VALID_VALUES)
    return num.where(valid, other=float(DGN_DEFAULT)).astype("Float64")


def recode_dms(pb190: pd.Series, idpartner: pd.Series) -> pd.Series:
    num = pd.to_numeric(pb190, errors="coerce").astype("Int64")
    recoded = num.map(DMS_RECODE)
    # individuals without marital status and no partner → single
    no_status = recoded.isna()
    no_partner = pd.to_numeric(idpartner, errors="coerce").fillna(0).eq(0)
    recoded = recoded.where(~(no_status & no_partner), other=float(DMS_DEFAULT))
    # individuals without marital status but with a partner → also single
    recoded = recoded.fillna(float(DMS_DEFAULT))
    return recoded.astype("Float64")


def recode_deh(pe040: pd.Series) -> pd.Series:
    num = pd.to_numeric(pe040, errors="coerce")
    result = pd.Series(float(DEH_DEFAULT), index=pe040.index, dtype="Float64")
    for low, high, target in DEH_RECODE_BOUNDARIES:
        mask = num.between(low, high, inclusive="both")
        result = result.where(~mask, other=float(target))
    return result


def recode_ddi(pl031: pd.Series, pb030: pd.Series) -> pd.Series:
    pl031_num = pd.to_numeric(pl031, errors="coerce")
    pb030_num = pd.to_numeric(pb030, errors="coerce")

    result = pd.Series(np.nan, index=pl031.index, dtype="Float64")
    result = result.where(~pl031_num.eq(8), other=float(DDI_DISABLED))
    not_disabled = pl031_num.notna() & ~pl031_num.eq(8)
    result = result.where(~not_disabled, other=float(DDI_NOT_DISABLED))
    not_applicable = pl031_num.isna() & pb030_num.isna()
    result = result.where(~not_applicable, other=float(DDI_NOT_APPLICABLE))
    return result

def compute_dag(rb080: pd.Series, rb010: pd.Series) -> pd.Series:
    birth_year = pd.to_numeric(rb080, errors="coerce")
    survey_year = pd.to_numeric(rb010, errors="coerce")
    age = survey_year - birth_year - 1
    return age.clip(lower=0).astype("Float64")


def compute_oecd_m(person_df: pd.DataFrame) -> pd.Series:
    """
    Compute OECD modified equivalence scale per household.

    Scale: 1.0 for the first adult, 0.5 for each additional adult,
    0.3 for each child under 14. Returns a Series indexed by household ID.

    Parameters
    ----------
    person_df : DataFrame with columns IDHH and DAG (uppercase).
    """
    age = pd.to_numeric(person_df["DAG"], errors="coerce")
    is_adult = age.ge(14)
    is_child = age.lt(14)

    grouped = person_df.assign(_adult=is_adult.astype(int), _child=is_child.astype(int))
    agg = grouped.groupby("IDHH")[["_adult", "_child"]].sum()

    oecd = 1.0 + (agg["_adult"] - 1).clip(lower=0) * 0.5 + agg["_child"] * 0.3
    return oecd.rename("oecd_m")

def recode_les(pl031: pd.Series, pl040: pd.Series, dag: pd.Series) -> pd.Series:
    pl031_num = pd.to_numeric(pl031, errors="coerce").astype("Int64")
    pl040_num = pd.to_numeric(pl040, errors="coerce").astype("Int64")
    age = pd.to_numeric(dag, errors="coerce")

    result = pd.Series(pd.NA, index=pl031.index, dtype="Int64")

    result = result.where(~age.lt(6), other=0)

    for pl031_val, les_val in PL031_TO_LES.items():
        mask = pl031_num.eq(pl031_val) & result.isna()
        result = result.where(~mask, other=les_val)

    for pl040_val, les_val in PL040_TO_LES.items():
        mask = pl040_num.eq(pl040_val) & result.isna()
        result = result.where(~mask, other=les_val)

    return result.fillna(LES_DEFAULT).astype("Float64")



def recode_lindi(pl111a: pd.Series) -> pd.Series:
    cleaned = pl111a.astype("string").str.strip().str.lower()
    return cleaned.map(LINDI_MAP).fillna(float(LINDI_DEFAULT)).astype("Float64")


def recode_amrtn(hh021: pd.Series) -> pd.Series:
    num = pd.to_numeric(hh021, errors="coerce").astype("Int64")
    result = num.copy().astype("Int64")
    result = result.where(~num.eq(1), other=2)
    result = result.where(~num.eq(2), other=1)
    result = result.where(~num.eq(5), other=6)
    return result.astype("Float64")


def scale_monthly(annual: pd.Series, hx010: pd.Series) -> pd.Series:
    """
    Convert annual gross income to monthly and scale by hx010.
    Mirrors EUROMOD formula: yem = py010g / 12 * hx010.
    """
    ann = pd.to_numeric(annual, errors="coerce").fillna(0.0)
    scale = pd.to_numeric(hx010, errors="coerce").fillna(1.0)
    return (ann / 12.0 * scale).astype("Float64")


def scale_monthly_hh(annual: pd.Series, hx010: pd.Series) -> pd.Series:
    """
    Convert annual household-level income to monthly, split by hx010
    and assigned to the default income recipient (household respondent).
    Mirrors EUROMOD formula: yds = hy020 / 12 * hx010.
    """
    ann = pd.to_numeric(annual, errors="coerce").fillna(0.0)
    scale = pd.to_numeric(hx010, errors="coerce").fillna(1.0)
    return (ann / 12.0 * scale).astype("Float64")


def compute_liwmy(pl073: pd.Series, pl074: pd.Series,
                  pl075: pd.Series, pl076: pd.Series) -> pd.Series:
    cols = [pd.to_numeric(s, errors="coerce") for s in (pl073, pl074, pl075, pl076)]
    total = pd.concat(cols, axis=1).sum(axis=1, min_count=1)
    return total.clip(upper=12).fillna(0.0).astype("Float64")


def compute_liwftmy(pl073: pd.Series, pl075: pd.Series) -> pd.Series:
    cols = [pd.to_numeric(s, errors="coerce") for s in (pl073, pl075)]
    total = pd.concat(cols, axis=1).sum(axis=1, min_count=1)
    return total.clip(upper=12).fillna(0.0).astype("Float64")


def compute_liwptmy(pl074: pd.Series, pl076: pd.Series) -> pd.Series:
    cols = [pd.to_numeric(s, errors="coerce") for s in (pl074, pl076)]
    total = pd.concat(cols, axis=1).sum(axis=1, min_count=1)
    return total.clip(upper=12).fillna(0.0).astype("Float64")


def compute_liwwh(pl200: pd.Series) -> pd.Series:
    years = pd.to_numeric(pl200, errors="coerce")
    months = (years * 12).clip(lower=0, upper=780)
    return months.fillna(0.0).astype("Float64")


def fill_zero(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype("Float64")
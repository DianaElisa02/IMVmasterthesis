"""Event-study diagnostics for continuous and tercile IMV exposure models."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
from scipy.stats import t as t_dist

from src.binned_did import compute_tercile_assignments
from src.constants import ANALYSIS_OUTCOMES, BALANCE_CONTROLS, EVENT_STUDY_REFERENCE_YEAR, EVENT_STUDY_YEARS

logger = logging.getLogger(__name__)
_CATEGORICAL_CONTROLS = {"head_age_group", "head_sex", "head_labour_group"}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _CATEGORICAL_CONTROLS:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def _wald_joint(fit, terms: list[str]) -> tuple[float, float]:
    names = fit.coef().index.tolist()
    selected = [term for term in terms if term in names]
    if not selected:
        return np.nan, np.nan
    restriction = np.zeros((len(selected), len(names)))
    for i, term in enumerate(selected):
        restriction[i, names.index(term)] = 1.0
    try:
        test = fit.wald_test(R=restriction)
        return float(test["statistic"]), float(test["pvalue"])
    except Exception as exc:
        logger.warning("Joint Wald test failed: %s", exc)
        return np.nan, np.nan


def _tidy(fit, terms: list[str], outcome: str, exposure_spec: str, model: str, reference_year: int, group: str | None = None) -> pd.DataFrame:
    n_clusters = int(fit._data["drgn2"].nunique()) if hasattr(fit, "_data") else np.nan
    crit = float(t_dist.ppf(0.975, df=max(int(n_clusters) - 1, 1))) if not pd.isna(n_clusters) else 1.96
    rows = []
    for term in terms:
        if term not in fit.coef().index:
            continue
        year = int(term.rsplit("_", 1)[-1])
        coef, se = float(fit.coef()[term]), float(fit.se()[term])
        rows.append({
            "model": model,
            "exposure_spec": exposure_spec,
            "outcome": outcome,
            "group": group,
            "year": year,
            "rel_year": year - reference_year,
            "term": term,
            "coef": coef,
            "se": se,
            "ci_low": coef - crit * se,
            "ci_high": coef + crit * se,
            "pval_crv1": float(fit.pvalue()[term]),
            "reference_year": reference_year,
            "pre_period": year < reference_year,
            "n_obs": int(fit._N) if hasattr(fit, "_N") else np.nan,
            "n_clusters": n_clusters,
        })
    return pd.DataFrame(rows)


def run_continuous_event_study(panel: pl.DataFrame, exposure_spec: str, outcomes: list[str] | None = None, reference_year: int = EVENT_STUDY_REFERENCE_YEAR) -> pd.DataFrame:
    if exposure_spec not in panel.columns:
        raise ValueError(f"Exposure '{exposure_spec}' not found")
    outcomes = outcomes or ANALYSIS_OUTCOMES
    years = [year for year in EVENT_STUDY_YEARS if year != reference_year]
    did = panel.filter(pl.col("year").is_in(EVENT_STUDY_YEARS + [reference_year]))
    terms = []
    for year in years:
        term = f"event_x_{year}"
        did = did.with_columns((pl.col("year").eq(year).cast(pl.Float64) * pl.col(exposure_spec)).alias(term))
        terms.append(term)
    df = did.to_pandas()
    controls = [col for col in BALANCE_CONTROLS if col in df.columns]
    logger.info("Continuous event-study controls [%s]: %s", exposure_spec, controls)
    frames = []
    for outcome in outcomes:
        required = [outcome, "drgn2", "year"] + terms + controls
        clean = _prepare(df[required].dropna())
        ctrl = (" + " + " + ".join(controls)) if controls else ""
        fit = pf.feols(f"{outcome} ~ {' + '.join(terms)}{ctrl} | drgn2 + year", data=clean, vcov={"CRV1": "drgn2"})
        result = _tidy(fit, terms, outcome, exposure_spec, "continuous", reference_year)
        pre_terms = [term for term in terms if int(term.rsplit('_', 1)[-1]) < reference_year]
        stat, pvalue = _wald_joint(fit, pre_terms)
        result["pretrend_wald_stat"] = stat
        result["pretrend_wald_p"] = pvalue
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_tercile_event_study(panel: pl.DataFrame, exposure_spec: str, outcomes: list[str] | None = None, reference_year: int = EVENT_STUDY_REFERENCE_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = outcomes or ANALYSIS_OUTCOMES
    assignments = compute_tercile_assignments(panel, exposure_spec)
    did = panel.join(pl.from_pandas(assignments[["drgn2", "exposure_tercile"]]), on="drgn2", how="inner")
    did = did.filter(pl.col("year").is_in(EVENT_STUDY_YEARS + [reference_year]))
    terms_by_group: dict[str, list[str]] = {"medium": [], "high": []}
    for group in ["medium", "high"]:
        indicator = pl.col("exposure_tercile").eq(group).cast(pl.Float64)
        for year in EVENT_STUDY_YEARS:
            if year == reference_year:
                continue
            term = f"event_{group}_{year}"
            did = did.with_columns((indicator * pl.col("year").eq(year).cast(pl.Float64)).alias(term))
            terms_by_group[group].append(term)
    all_terms = terms_by_group["medium"] + terms_by_group["high"]
    df = did.to_pandas()
    controls = [col for col in BALANCE_CONTROLS if col in df.columns]
    logger.info("Tercile event-study controls [%s]: %s", exposure_spec, controls)
    frames = []
    for outcome in outcomes:
        required = [outcome, "drgn2", "year"] + all_terms + controls
        clean = _prepare(df[required].dropna())
        ctrl = (" + " + " + ".join(controls)) if controls else ""
        fit = pf.feols(f"{outcome} ~ {' + '.join(all_terms)}{ctrl} | drgn2 + year", data=clean, vcov={"CRV1": "drgn2"})
        outcome_frames = []
        all_pre_terms = []
        for group, terms in terms_by_group.items():
            result = _tidy(fit, terms, outcome, exposure_spec, "tercile", reference_year, group)
            pre_terms = [term for term in terms if int(term.rsplit('_', 1)[-1]) < reference_year]
            all_pre_terms.extend(pre_terms)
            stat, pvalue = _wald_joint(fit, pre_terms)
            result["pretrend_group_wald_stat"] = stat
            result["pretrend_group_wald_p"] = pvalue
            outcome_frames.append(result)
        stat, pvalue = _wald_joint(fit, all_pre_terms)
        for result in outcome_frames:
            result["pretrend_wald_stat"] = stat
            result["pretrend_wald_p"] = pvalue
            frames.append(result)
    assignments["exposure_spec"] = exposure_spec
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), assignments

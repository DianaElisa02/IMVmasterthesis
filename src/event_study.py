"""
event_study.py
==============
Validity and robustness checks for the IMV DiD analysis.

This module implements continuous event-study and placebo specifications used
for the main robustness checks. Tercile functions are kept for appendix work,
but the preferred runner is intentionally parsimonious because treatment
variation occurs across a small number of regional clusters.

Important interpretation:
Event-study and placebo tests are diagnostics. They can provide evidence
consistent or inconsistent with parallel pre-trends, but they do not prove the
parallel trends assumption.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
from scipy.stats import t as t_dist

from src.constants import (
    BALANCE_CONTROLS,
    EVENT_STUDY_REFERENCE_YEAR,
    EVENT_STUDY_YEARS,
    EXPOSURE_SPECS,
    PLACEBO_FAKE_TREATMENT_YEAR,
    PLACEBO_YEARS,
    YEARS,
)

logger = logging.getLogger(__name__)

PRIMARY_SPEC = EXPOSURE_SPECS[0]
REF_YEAR = EVENT_STUDY_REFERENCE_YEAR
PRE_YEARS = YEARS
EVENT_YEARS = EVENT_STUDY_YEARS
ALL_EVENT_SAMPLE_YEARS = sorted(set(PRE_YEARS + EVENT_YEARS + [REF_YEAR]))


# =============================================================================
# Utilities
# =============================================================================
def _safe_float(value) -> float:
    try:
        if hasattr(value, "iloc"):
            return float(value.iloc[0])
        return float(value)
    except Exception:
        return np.nan


def _extract_boot_pvalue(boot) -> float:
    """Extract p-value from PyFixest wildboottest output across versions."""
    for key in ["Pr(>|t|)", "p-value", "pvalue", "p_value"]:
        try:
            if key in boot:
                return _safe_float(boot[key])
        except Exception:
            continue
    return np.nan


def _available_controls(df: pd.DataFrame, controls: list[str] | None = None) -> list[str]:
    if controls is None:
        controls = BALANCE_CONTROLS
    return [c for c in controls if c in df.columns]


def _tcrit_from_clusters(df: pd.DataFrame, cluster_col: str = "drgn2") -> float:
    n_clusters = int(df[cluster_col].nunique())
    if n_clusters <= 1:
        return 1.96
    return float(t_dist.ppf(0.975, df=n_clusters - 1))


def _run_wcb(fit, param: str, seed: int, reps: int = 9999) -> float:
    try:
        boot = fit.wildboottest(param=param, reps=reps, seed=seed)
        return _extract_boot_pvalue(boot)
    except Exception as exc:
        logger.warning("WCB failed for %s: %s", param, exc)
        return np.nan


def _cluster_count(df: pd.DataFrame) -> int:
    return int(df["drgn2"].nunique())


def _region_list(df: pd.DataFrame) -> list[int]:
    return sorted([int(x) for x in df["drgn2"].dropna().unique().tolist()])


def _wald_joint_test(fit, terms: list[str]) -> tuple[float, float]:
    """
    Cluster-robust joint Wald diagnostic for selected terms.

    This is not a wild-bootstrap joint test and should not be interpreted as a
    definitive pass/fail test of parallel trends.
    """
    coef_names = fit.coef().index.tolist()
    present_terms = [t for t in terms if t in coef_names]

    if not present_terms:
        return np.nan, np.nan

    R = np.zeros((len(present_terms), len(coef_names)))
    for i, term in enumerate(present_terms):
        R[i, coef_names.index(term)] = 1.0

    try:
        wald = fit.wald_test(R=R)
        return float(wald["statistic"]), float(wald["pvalue"])
    except Exception as exc:
        logger.warning("Joint Wald diagnostic failed for terms %s: %s", present_terms, exc)
        return np.nan, np.nan


# =============================================================================
# Tercile utilities retained for appendix checks
# =============================================================================
def make_region_terciles(
    panel: pl.DataFrame,
    exposure: str,
    region_col: str = "drgn2",
) -> pd.DataFrame:
    """Compute exposure terciles automatically from region-level exposure values."""
    if exposure not in panel.columns:
        raise ValueError(f"Exposure variable '{exposure}' not found in panel.")

    df = (
        panel
        .select([region_col, exposure])
        .drop_nulls([region_col, exposure])
        .group_by(region_col)
        .agg(pl.col(exposure).mean().alias("exposure_value"))
        .sort("exposure_value")
        .to_pandas()
    )

    if df.empty:
        raise ValueError(f"No non-missing region exposure values for '{exposure}'.")

    n_regions = len(df)
    if n_regions < 3:
        raise ValueError(f"Need at least 3 regions to form terciles; found {n_regions}.")

    df["_rank"] = df["exposure_value"].rank(method="first")
    df["exposure_tercile"] = pd.qcut(
        df["_rank"], q=3, labels=["low", "medium", "high"]
    ).astype(str)

    df["medium_exp"] = (df["exposure_tercile"] == "medium").astype(float)
    df["high_exp"] = (df["exposure_tercile"] == "high").astype(float)

    df = df.drop(columns=["_rank"]).sort_values(["exposure_tercile", "exposure_value"])

    logger.info(
        "Terciles for %s: low=%d, medium=%d, high=%d",
        exposure,
        int((df["exposure_tercile"] == "low").sum()),
        int((df["exposure_tercile"] == "medium").sum()),
        int((df["exposure_tercile"] == "high").sum()),
    )

    return df


def attach_terciles(
    panel: pl.DataFrame,
    exposure: str,
    region_col: str = "drgn2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert panel to pandas and attach automatically computed exposure terciles."""
    terciles = make_region_terciles(panel, exposure, region_col=region_col)
    df = panel.to_pandas()
    df = df.merge(
        terciles[[region_col, "exposure_value", "exposure_tercile", "medium_exp", "high_exp"]],
        on=region_col,
        how="left",
        validate="many_to_one",
    )
    return df, terciles


def build_event_study_data(
    panel: pl.DataFrame,
    exposure: str = PRIMARY_SPEC,
) -> pd.DataFrame:
    """
    Build event-study data for a continuous exposure specification.

    Reference year is 2019. The interaction terms are created for all years in
    EVENT_STUDY_YEARS, which should exclude 2019 and 2020.
    """
    if exposure not in panel.columns:
        raise ValueError(f"Exposure variable '{exposure}' not found in panel.")

    keep_years = sorted(set(PRE_YEARS + EVENT_YEARS + [REF_YEAR]))

    df = panel.filter(pl.col("year").is_in(keep_years)).to_pandas()
    df["_exposure"] = df[exposure]

    for yr in EVENT_YEARS:
        df[f"yr_{yr}"] = (df["year"] == yr).astype(float)
        df[f"yr_{yr}_x_exp"] = df[f"yr_{yr}"] * df["_exposure"]

    logger.info(
        "Continuous event-study data built: exposure=%s | obs=%d | years=%s | clusters=%d",
        exposure,
        len(df),
        sorted(df["year"].dropna().unique().tolist()),
        _cluster_count(df),
    )

    return df


def run_event_study(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str] | None = None,
    exposure: str = PRIMARY_SPEC,
    region_trends: bool = False,
    model: str = "continuous_event_study",
    reps: int = 9999,
    seed_base: int = 42,
    run_wcb: bool = True,
) -> pd.DataFrame:
    """Estimate continuous event-study model for one outcome and one exposure."""
    controls = _available_controls(df, controls)
    interaction_terms = [f"yr_{yr}_x_exp" for yr in EVENT_YEARS]

    work = df.copy()
    trend_terms: list[str] = []

    if region_trends:
        work["year_c"] = work["year"] - REF_YEAR
        regions = _region_list(work)
        omitted_regions = regions[:2]
        regions_for_trends = regions[2:]

        for reg in regions_for_trends:
            col = f"trend_r{reg}"
            work[col] = (work["drgn2"].astype(int) == reg).astype(float) * work["year_c"]
            trend_terms.append(col)

        logger.info(
            "Added region-specific trend deviations: %d terms; omitted regions=%s",
            len(trend_terms),
            omitted_regions,
        )

    ctrl_str = (" + " + " + ".join(controls)) if controls else ""
    trend_str = (" + " + " + ".join(trend_terms)) if trend_terms else ""

    formula = (
        f"{outcome} ~ "
        + " + ".join(interaction_terms)
        + ctrl_str
        + trend_str
        + " | drgn2 + year"
    )

    keep_cols = [outcome, "drgn2", "year"] + interaction_terms + controls + trend_terms
    work = work[[c for c in keep_cols if c in work.columns]].dropna().reset_index(drop=True)

    if work.empty:
        raise ValueError(f"No complete cases for outcome={outcome}, exposure={exposure}.")

    fit = pf.feols(formula, data=work, vcov={"CRV1": "drgn2"})

    tcrit = _tcrit_from_clusters(work)
    n_clusters = _cluster_count(work)
    n_obs = len(work)

    rows = []
    for yr in EVENT_YEARS:
        term = f"yr_{yr}_x_exp"
        coef = float(fit.coef().get(term, np.nan))
        se = float(fit.se().get(term, np.nan))
        p_crv1 = float(fit.pvalue().get(term, np.nan))
        p_wbt = _run_wcb(fit, term, seed=seed_base + yr, reps=reps) if run_wcb else np.nan

        rows.append({
            "model": model,
            "exposure_spec": exposure,
            "outcome": outcome,
            "year": yr,
            "rel_year": yr - REF_YEAR,
            "term": term,
            "coef": coef,
            "se": se,
            "ci_low": coef - tcrit * se,
            "ci_high": coef + tcrit * se,
            "pval_crv1": p_crv1,
            "pval_wbt": p_wbt,
            "pre_period": yr < REF_YEAR,
            "reference_year": REF_YEAR,
            "n_obs": n_obs,
            "n_clusters": n_clusters,
        })

    rows.append({
        "model": model,
        "exposure_spec": exposure,
        "outcome": outcome,
        "year": REF_YEAR,
        "rel_year": 0,
        "term": "reference",
        "coef": 0.0,
        "se": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "pval_crv1": np.nan,
        "pval_wbt": np.nan,
        "pre_period": True,
        "reference_year": REF_YEAR,
        "n_obs": n_obs,
        "n_clusters": n_clusters,
    })

    result = pd.DataFrame(rows).sort_values(["year"]).reset_index(drop=True)

    pre_terms = [f"yr_{yr}_x_exp" for yr in EVENT_YEARS if yr < REF_YEAR]
    post_terms = [f"yr_{yr}_x_exp" for yr in EVENT_YEARS if yr > REF_YEAR]
    wald_stat, wald_p = _wald_joint_test(fit, pre_terms)

    lead_coefs = [
        float(fit.coef().get(t, np.nan))
        for t in pre_terms
        if t in fit.coef().index
    ]
    lead_coefs = [x for x in lead_coefs if not np.isnan(x)]

    post_coefs = [
        float(fit.coef().get(t, np.nan))
        for t in post_terms
        if t in fit.coef().index
    ]
    post_coefs = [x for x in post_coefs if not np.isnan(x)]

    lead_mean = float(np.mean(lead_coefs)) if lead_coefs else np.nan
    lead_max_abs = float(np.max(np.abs(lead_coefs))) if lead_coefs else np.nan
    post_mean = float(np.mean(post_coefs)) if post_coefs else np.nan
    post_max_abs = float(np.max(np.abs(post_coefs))) if post_coefs else np.nan
    lead_to_post_ratio = (
        lead_max_abs / post_max_abs
        if not np.isnan(lead_max_abs) and not np.isnan(post_max_abs) and post_max_abs > 0
        else np.nan
    )

    result["pretrend_wald_stat"] = wald_stat
    result["pretrend_wald_p"] = wald_p
    result["lead_mean"] = lead_mean
    result["lead_max_abs"] = lead_max_abs
    result["post_mean"] = post_mean
    result["post_max_abs"] = post_max_abs
    result["lead_to_post_ratio"] = lead_to_post_ratio
    result["region_trends"] = bool(region_trends)

    return result


def run_placebo_continuous(
    panel: pl.DataFrame,
    outcome: str,
    exposure: str = PRIMARY_SPEC,
    controls: list[str] | None = None,
    reps: int = 9999,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate continuous placebo DiD using only 2017-2019."""
    if exposure not in panel.columns:
        raise ValueError(f"Exposure variable '{exposure}' not found in panel.")

    df = panel.filter(pl.col("year").is_in(PLACEBO_YEARS)).to_pandas()
    controls = _available_controls(df, controls)

    df["_exposure"] = df[exposure]
    df["post_fake"] = (df["year"] == PLACEBO_FAKE_TREATMENT_YEAR).astype(float)
    df["post_fake_x_exp"] = df["post_fake"] * df["_exposure"]

    ctrl_str = (" + " + " + ".join(controls)) if controls else ""
    formula = f"{outcome} ~ post_fake_x_exp{ctrl_str} | drgn2 + year"

    keep_cols = [outcome, "drgn2", "year", "post_fake_x_exp"] + controls
    df_clean = df[[c for c in keep_cols if c in df.columns]].dropna().reset_index(drop=True)

    if df_clean.empty:
        raise ValueError(f"No complete cases for continuous placebo: {outcome}, {exposure}")

    fit = pf.feols(formula, data=df_clean, vcov={"CRV1": "drgn2"})

    term = "post_fake_x_exp"
    coef = float(fit.coef().get(term, np.nan))
    se = float(fit.se().get(term, np.nan))
    p_crv1 = float(fit.pvalue().get(term, np.nan))
    p_wbt = _run_wcb(fit, term, seed=seed, reps=reps)
    tcrit = _tcrit_from_clusters(df_clean)

    return pd.DataFrame([{
        "model": "continuous_placebo",
        "exposure_spec": exposure,
        "outcome": outcome,
        "term": term,
        "coef": coef,
        "se": se,
        "ci_low": coef - tcrit * se,
        "ci_high": coef + tcrit * se,
        "pval_crv1": p_crv1,
        "pval_wbt": p_wbt,
        "fake_treatment_year": PLACEBO_FAKE_TREATMENT_YEAR,
        "n_obs": len(df_clean),
        "n_clusters": _cluster_count(df_clean),
        "diagnostic_flag": (
            "placebo_not_statistically_distinguishable_from_zero"
            if not np.isnan(p_wbt) and p_wbt > 0.10
            else "placebo_statistically_distinguishable_from_zero"
            if not np.isnan(p_wbt)
            else "wcb_unavailable"
        ),
    }])


# =============================================================================
# Tercile appendix checks
# =============================================================================
def build_tercile_event_study_data(
    panel: pl.DataFrame,
    exposure: str = PRIMARY_SPEC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build event-study data using exposure terciles for appendix checks."""
    keep_years = sorted(set(PRE_YEARS + EVENT_YEARS + [REF_YEAR]))
    panel_keep = panel.filter(pl.col("year").is_in(keep_years))
    df, terciles = attach_terciles(panel_keep, exposure=exposure)

    for yr in EVENT_YEARS:
        df[f"yr_{yr}"] = (df["year"] == yr).astype(float)
        df[f"yr_{yr}_x_medium"] = df[f"yr_{yr}"] * df["medium_exp"]
        df[f"yr_{yr}_x_high"] = df[f"yr_{yr}"] * df["high_exp"]

    logger.info(
        "Tercile event-study data built: exposure=%s | obs=%d | years=%s | clusters=%d",
        exposure,
        len(df),
        sorted(df["year"].dropna().unique().tolist()),
        _cluster_count(df),
    )
    return df, terciles


def run_event_study_terciles(
    df: pd.DataFrame,
    outcome: str,
    exposure: str = PRIMARY_SPEC,
    controls: list[str] | None = None,
    reps: int = 9999,
    seed_base: int = 1000,
    run_wcb: bool = True,
) -> pd.DataFrame:
    """Estimate tercile event-study model for appendix checks."""
    controls = _available_controls(df, controls)

    interaction_terms = []
    for yr in EVENT_YEARS:
        interaction_terms.extend([f"yr_{yr}_x_medium", f"yr_{yr}_x_high"])

    ctrl_str = (" + " + " + ".join(controls)) if controls else ""
    formula = f"{outcome} ~ " + " + ".join(interaction_terms) + ctrl_str + " | drgn2 + year"

    keep_cols = [outcome, "drgn2", "year"] + interaction_terms + controls
    work = df[[c for c in keep_cols if c in df.columns]].dropna().reset_index(drop=True)

    if work.empty:
        raise ValueError(f"No complete cases for tercile event study: {outcome}, {exposure}")

    fit = pf.feols(formula, data=work, vcov={"CRV1": "drgn2"})
    tcrit = _tcrit_from_clusters(work)
    n_clusters = _cluster_count(work)
    n_obs = len(work)

    rows = []
    for yr in EVENT_YEARS:
        for group in ["medium", "high"]:
            term = f"yr_{yr}_x_{group}"
            coef = float(fit.coef().get(term, np.nan))
            se = float(fit.se().get(term, np.nan))
            p_crv1 = float(fit.pvalue().get(term, np.nan))
            p_wbt = _run_wcb(fit, term, seed=seed_base + yr + (10 if group == "high" else 0), reps=reps) if run_wcb else np.nan
            rows.append({
                "model": "tercile_event_study",
                "exposure_spec": exposure,
                "outcome": outcome,
                "group": group,
                "year": yr,
                "rel_year": yr - REF_YEAR,
                "term": term,
                "coef": coef,
                "se": se,
                "ci_low": coef - tcrit * se,
                "ci_high": coef + tcrit * se,
                "pval_crv1": p_crv1,
                "pval_wbt": p_wbt,
                "pre_period": yr < REF_YEAR,
                "reference_year": REF_YEAR,
                "n_obs": n_obs,
                "n_clusters": n_clusters,
            })

    for group in ["medium", "high"]:
        rows.append({
            "model": "tercile_event_study",
            "exposure_spec": exposure,
            "outcome": outcome,
            "group": group,
            "year": REF_YEAR,
            "rel_year": 0,
            "term": "reference",
            "coef": 0.0,
            "se": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "pval_crv1": np.nan,
            "pval_wbt": np.nan,
            "pre_period": True,
            "reference_year": REF_YEAR,
            "n_obs": n_obs,
            "n_clusters": n_clusters,
        })

    result = pd.DataFrame(rows).sort_values(["group", "year"]).reset_index(drop=True)

    pre_terms = []
    for yr in EVENT_YEARS:
        if yr < REF_YEAR:
            pre_terms.extend([f"yr_{yr}_x_medium", f"yr_{yr}_x_high"])

    wald_stat, wald_p = _wald_joint_test(fit, pre_terms)
    result["pretrend_wald_stat"] = wald_stat
    result["pretrend_wald_p"] = wald_p
    return result


def run_placebo_terciles(
    panel: pl.DataFrame,
    outcome: str,
    exposure: str = PRIMARY_SPEC,
    controls: list[str] | None = None,
    reps: int = 9999,
    seed_base: int = 2000,
) -> pd.DataFrame:
    """Estimate placebo DiD using exposure terciles for appendix checks."""
    panel_pre = panel.filter(pl.col("year").is_in(PLACEBO_YEARS))
    df, _terciles = attach_terciles(panel_pre, exposure=exposure)
    controls = _available_controls(df, controls)

    df["post_fake"] = (df["year"] == PLACEBO_FAKE_TREATMENT_YEAR).astype(float)
    df["post_fake_x_medium"] = df["post_fake"] * df["medium_exp"]
    df["post_fake_x_high"] = df["post_fake"] * df["high_exp"]

    terms = ["post_fake_x_medium", "post_fake_x_high"]
    ctrl_str = (" + " + " + ".join(controls)) if controls else ""
    formula = f"{outcome} ~ " + " + ".join(terms) + ctrl_str + " | drgn2 + year"

    keep_cols = [outcome, "drgn2", "year"] + terms + controls
    work = df[[c for c in keep_cols if c in df.columns]].dropna().reset_index(drop=True)

    if work.empty:
        raise ValueError(f"No complete cases for tercile placebo: {outcome}, {exposure}")

    fit = pf.feols(formula, data=work, vcov={"CRV1": "drgn2"})
    tcrit = _tcrit_from_clusters(work)

    rows = []

    for idx, term in enumerate(terms):
        coef = float(fit.coef().get(term, np.nan))
        se = float(fit.se().get(term, np.nan))
        p_crv1 = float(fit.pvalue().get(term, np.nan))
        p_wbt = _run_wcb(fit, term, seed=seed_base + idx, reps=reps)
        group = "medium" if term.endswith("medium") else "high"
        rows.append({
            "model": "tercile_placebo",
            "exposure_spec": exposure,
            "outcome": outcome,
            "group": group,
            "term": term,
            "coef": coef,
            "se": se,
            "ci_low": coef - tcrit * se,
            "ci_high": coef + tcrit * se,
            "pval_crv1": p_crv1,
            "pval_wbt": p_wbt,
            "fake_treatment_year": PLACEBO_FAKE_TREATMENT_YEAR,
            "n_obs": len(work),
            "n_clusters": _cluster_count(work),
            "diagnostic_flag": (
                "placebo_not_statistically_distinguishable_from_zero"
                if not np.isnan(p_wbt) and p_wbt > 0.10
                else "placebo_statistically_distinguishable_from_zero"
                if not np.isnan(p_wbt)
                else "wcb_unavailable"
            ),
        })

    wald_stat, wald_p = _wald_joint_test(fit, terms)
    result = pd.DataFrame(rows)
    result["joint_wald_stat"] = wald_stat
    result["joint_wald_p"] = wald_p
    return result

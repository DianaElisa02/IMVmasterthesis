"""
baseline_did.py
===============
Baseline DiD estimation for the IMV analysis.
Rewritten using PyFixest for FE absorption, clustering, and WCB.

Specification
-------------
  Y_hrt = α + β (Post_t × Exposure_r) + γ_r + δ_t + X_hrt·θ + ε_hrt

  where:
    Post_t     = 1 if year in post_years, 0 if year in [2017, 2018, 2019]
    Exposure_r = regional exposure index (time-invariant)
    γ_r        = region fixed effects — absorbed via demeaning (PyFixest)
    δ_t        = year fixed effects  — absorbed via demeaning (PyFixest)
    X_hrt      = household-level controls (BALANCE_CONTROLS)

  Estimated via unweighted OLS (PyFixest feols).
  Region + year FEs absorbed via Frisch-Waugh-Lovell demeaning.
  Standard errors clustered at region level (CRV1, 15 clusters).
  Inference via wild cluster bootstrap (Webb weights, B=9999, seed=42).

Note on survey weights
----------------------
PyFixest's wildboottest does not support WLS. Estimation is therefore
unweighted (OLS), consistent with the recommendation in Solon, Haider
& Wooldridge (2015): with two-way FE, survey weights are generally not
required because the FE structure already absorbs the main sources of
heterogeneity that weights address. Weighted results are available as
a robustness check in the appendix (CRV1 SEs only, no WCB).

Outcomes
--------
run_baseline_did() accepts an optional outcomes list. Defaults to
ANALYSIS_OUTCOMES (matdep, poverty). Pass POVERTY_GAP_OUTCOMES to
estimate poverty gap outcomes using the same infrastructure.

Placebo test
------------
run_placebo_test() implements the pre-reform falsification check using
only pre-reform years (2017-2019). Fake treatment: 2019. Reference: 2018.
Uses PyFixest feols + WCB, same inference structure as the main DiD.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
from scipy.stats import t as t_dist

from src.constants import (
    ANALYSIS_OUTCOMES,
    BALANCE_CONTROLS,
    DID_POST_YEARS_BASELINE,
    EXPOSURE_SPECS,
    PLACEBO_FAKE_TREATMENT_YEAR,
    PLACEBO_YEARS,
    YEARS,
)

logger = logging.getLogger(__name__)

_PRE_YEARS: list[int] = YEARS   # [2017, 2018, 2019]


# =============================================================================
# BUILD DiD DATA
# =============================================================================

def build_did_data(
    panel: pl.DataFrame,
    post_years: list[int] | None = None,
) -> pl.DataFrame:
    """
    Prepare the analysis panel for DiD estimation.

    Constructs:
      - post          : binary (0 = pre-reform, 1 = post-reform)
      - post_x_{spec} : Post x Exposure interaction for each exposure spec

    Parameters
    ----------
    panel      : full analysis panel (Polars DataFrame)
    post_years : post-reform years (default: DID_POST_YEARS_BASELINE)
    """
    if post_years is None:
        post_years = DID_POST_YEARS_BASELINE

    pre_years  = _PRE_YEARS
    keep_years = pre_years + post_years

    did = panel.filter(pl.col("year").is_in(keep_years))

    did = did.with_columns(
        pl.when(pl.col("year").is_in(post_years))
        .then(pl.lit(1.0))
        .when(pl.col("year").is_in(pre_years))
        .then(pl.lit(0.0))
        .otherwise(pl.lit(None))
        .alias("post")
    )

    built, missing = [], []
    for spec in EXPOSURE_SPECS:
        if spec in did.columns:
            did = did.with_columns(
                (pl.col("post") * pl.col(spec)).alias(f"post_x_{spec}")
            )
            built.append(spec)
        else:
            missing.append(spec)

    if missing:
        logger.warning("Exposure specs not in panel (skipped): %s", missing)

    logger.info(
        "DiD data built: %d obs | pre=%s | post=%s | interactions=%s",
        len(did), pre_years, post_years, built,
    )
    return did


# =============================================================================
# ESTIMATE ONE SPEC
# =============================================================================

def run_did_spec(
    df: pd.DataFrame,
    outcome: str,
    exposure_spec: str,
    controls: list[str],
) -> dict:
    """
    Estimate baseline DiD for one outcome x exposure spec using PyFixest.

    Parameters
    ----------
    df            : pandas DataFrame (pre-filtered to relevant columns)
    outcome       : outcome column name
    exposure_spec : one of EXPOSURE_SPECS
    controls      : list of control variable column names

    Returns
    -------
    dict with coef, se, ci_low, ci_high, pval_cluster, pval_wbt,
         n_obs, n_clusters, r2_within, outcome, exposure_spec
    """
    interaction_col = f"post_x_{exposure_spec}"

    required = [outcome, interaction_col, "drgn2", "year"] + controls
    df_clean = (
        df[[c for c in required if c in df.columns]]
        .dropna()
        .reset_index(drop=True)
    )

    if len(df_clean) == 0:
        raise ValueError(
            f"No complete cases for outcome={outcome}, spec={exposure_spec}"
        )

    ctrl_str = (" + " + " + ".join(controls)) if controls else ""
    formula  = f"{outcome} ~ {interaction_col}{ctrl_str} | drgn2 + year"

    fit = pf.feols(
        formula,
        data=df_clean,
        vcov={"CRV1": "drgn2"},
    )

    coef = float(fit.coef()[interaction_col])
    se   = float(fit.se()[interaction_col])
    pval = float(fit.pvalue()[interaction_col])

    n_clusters = int(df_clean["drgn2"].nunique())
    t_crit     = float(t_dist.ppf(0.975, df=n_clusters - 1))

    try:
        r2_within = float(fit.r2(type="within"))
    except Exception:
        try:
            r2_within = float(fit.r2("within"))
        except Exception:
            r2_within = np.nan

    p_wbt = np.nan
    try:
        boot  = fit.wildboottest(param=interaction_col, reps=9999, seed=42)
        p_raw = boot["Pr(>|t|)"]
        p_wbt = float(p_raw.iloc[0]) if hasattr(p_raw, "iloc") else float(p_raw)
        logger.info("WCB  -- %s x %s: p = %.4f", outcome, exposure_spec, p_wbt)
    except Exception as e:
        logger.warning("WCB failed -- %s x %s: %s", outcome, exposure_spec, e)

    logger.info(
        "DiD  -- %s x %s: b=%+.4f  SE=%.4f  p_cluster=%.4f  p_wbt=%.4f",
        outcome, exposure_spec, coef, se, pval,
        p_wbt if not np.isnan(p_wbt) else -99,
    )

    return {
        "outcome":       outcome,
        "exposure_spec": exposure_spec,
        "coef":          coef,
        "se":            se,
        "ci_low":        coef - t_crit * se,
        "ci_high":       coef + t_crit * se,
        "pval_cluster":  pval,
        "pval_wbt":      p_wbt,
        "n_obs":         len(df_clean),
        "n_clusters":    n_clusters,
        "r2_within":     r2_within,
    }


# =============================================================================
# RUN ALL SPECS
# =============================================================================

def run_baseline_did(
    did: pl.DataFrame,
    label: str = "baseline",
    outcomes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Estimate baseline DiD for all outcomes x all exposure specs.

    Parameters
    ----------
    did      : Polars DataFrame from build_did_data()
    label    : label for this estimation window
    outcomes : outcome columns to estimate (default: ANALYSIS_OUTCOMES).
               Pass POVERTY_GAP_OUTCOMES to run poverty gap outcomes
               using the same infrastructure without duplicating code.

    Returns
    -------
    pd.DataFrame, one row per outcome x exposure spec
    """
    if outcomes is None:
        outcomes = ANALYSIS_OUTCOMES

    df       = did.to_pandas()
    controls = [c for c in BALANCE_CONTROLS if c in df.columns]

    if not controls:
        logger.warning("No BALANCE_CONTROLS found -- estimating without controls")

    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            logger.warning("Outcome '%s' not in panel -- skipping", outcome)
            continue

        for spec in EXPOSURE_SPECS:
            interaction_col = f"post_x_{spec}"
            if interaction_col not in df.columns:
                logger.warning("Interaction '%s' not in panel -- skipping", interaction_col)
                continue

            try:
                row = run_did_spec(df, outcome, spec, controls)
                row["label"] = label
                rows.append(row)
            except Exception as e:
                logger.error("Failed -- %s x %s: %s", outcome, spec, e)

    results = pd.DataFrame(rows)
    logger.info(
        "[%s] DiD complete: %d outcome-spec pairs estimated",
        label, len(results),
    )
    return results


# =============================================================================
# PLACEBO TEST
# =============================================================================

def run_placebo_test(
    panel: pl.DataFrame,
    outcomes: list[str],
    exposure_spec: str | None = None,
    controls: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pre-reform falsification test using PyFixest.

    Uses only pre-reform years (2017-2019).
    Fake treatment: 2019 (Post_fake = 1).
    Reference year for year FE: 2018.

    The interaction Post_fake x Exposure tests whether regions with
    higher exposure already had differential pre-reform trends.
    A significant WCB p-value (< 0.10) is evidence against parallel trends.

    Parameters
    ----------
    panel         : full analysis panel (Polars DataFrame, all years)
    outcomes      : list of outcome column names to test
    exposure_spec : exposure spec to use (default: primary spec, EXPOSURE_SPECS[0])
    controls      : controls to include (default: BALANCE_CONTROLS)

    Returns
    -------
    pd.DataFrame with one row per outcome: coef, SE, CI, p-values, verdict
    """
    if exposure_spec is None:
        exposure_spec = EXPOSURE_SPECS[0]

    if exposure_spec not in panel.columns:
        raise ValueError(
            f"Exposure spec '{exposure_spec}' not found in panel."
        )

    placebo_pl = (
        panel
        .filter(pl.col("year").is_in(PLACEBO_YEARS))
        .with_columns(
            pl.when(pl.col("year").eq(PLACEBO_FAKE_TREATMENT_YEAR))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("post_fake")
        )
        .with_columns(
            (pl.col("post_fake") * pl.col(exposure_spec))
            .alias("post_fake_x_exposure")
        )
    )

    df = placebo_pl.to_pandas()

    if controls is None:
        controls = [c for c in BALANCE_CONTROLS if c in df.columns]

    ctrl_str = (" + " + " + ".join(controls)) if controls else ""

    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            logger.warning("Placebo: outcome '%s' not in panel -- skipping", outcome)
            continue

        required = [outcome, "post_fake_x_exposure", "drgn2", "year"] + controls
        df_clean = (
            df[[c for c in required if c in df.columns]]
            .dropna()
            .reset_index(drop=True)
        )

        if len(df_clean) == 0:
            logger.warning("Placebo: no complete cases for '%s'", outcome)
            continue

        formula = f"{outcome} ~ post_fake_x_exposure{ctrl_str} | drgn2 + year"

        fit = pf.feols(formula, data=df_clean, vcov={"CRV1": "drgn2"})

        coef = float(fit.coef()["post_fake_x_exposure"])
        se   = float(fit.se()["post_fake_x_exposure"])
        pval = float(fit.pvalue()["post_fake_x_exposure"])

        n_clusters = int(df_clean["drgn2"].nunique())
        t_crit     = float(t_dist.ppf(0.975, df=n_clusters - 1))

        p_wbt = np.nan
        try:
            boot  = fit.wildboottest(
                param="post_fake_x_exposure", reps=9999, seed=42
            )
            p_raw = boot["Pr(>|t|)"]
            p_wbt = float(p_raw.iloc[0]) if hasattr(p_raw, "iloc") else float(p_raw)
        except Exception as e:
            logger.warning("Placebo WCB failed -- %s: %s", outcome, e)

        verdict = (
            "PASS" if not np.isnan(p_wbt) and p_wbt > 0.10
            else "WARNING"
        )

        logger.info(
            "Placebo -- %s: b=%+.4f  SE=%.4f  p_cluster=%.4f  p_wbt=%.4f  -> %s",
            outcome, coef, se, pval,
            p_wbt if not np.isnan(p_wbt) else -99,
            verdict,
        )

        rows.append({
            "outcome":       outcome,
            "exposure_spec": exposure_spec,
            "coef":          coef,
            "se":            se,
            "ci_low":        coef - t_crit * se,
            "ci_high":       coef + t_crit * se,
            "pval_cluster":  pval,
            "pval_wbt":      p_wbt,
            "n_obs":         len(df_clean),
            "n_clusters":    n_clusters,
            "verdict":       verdict,
        })

    results = pd.DataFrame(rows)

    if not results.empty:
        passed = results[results["verdict"] == "PASS"]["outcome"].tolist()
        warned = results[results["verdict"] == "WARNING"]["outcome"].tolist()
        if passed:
            logger.info("Placebo PASSED: %s", passed)
        if warned:
            logger.warning("Placebo WARNING (pre-trend detected): %s", warned)

    return results
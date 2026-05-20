"""
binned_did.py
=============
Binned DiD specification — relaxes the linear dose-response assumption
of the continuous TWFE by estimating separate ATTs for medium and high
exposure terciles relative to the low tercile (reference group).

Mirrors the structure of baseline_did.py exactly:
  - build_binned_did_data() : constructs post indicator + tercile interactions
  - run_binned_did_spec()   : estimates one outcome with PyFixest + WCB
  - run_binned_did()        : loops over outcomes, returns tidy DataFrame

Specification
-------------
  Y_hrt = α + β_M (Post_t × 1[medium_r])
              + β_H (Post_t × 1[high_r])
              + γ_r + δ_t + X_hrt·θ + ε_hrt

  Reference group: low-exposure tercile (most generous pre-reform RMI,
  smallest reform-induced change in protection).

Interpretation
--------------
  β_M : post-reform change in outcome for medium tercile vs low tercile
  β_H : post-reform change in outcome for high tercile vs low tercile
  β_H ≈ 2 × β_M : linear dose-response holds — continuous TWFE is valid
  β_H >> β_M    : effects concentrated at top — continuous TWFE dilutes
  Both ≈ 0      : null result genuine across the entire distribution

Note on tercile main effects and region FE
------------------------------------------
tercile_medium and tercile_high are time-invariant — regions never switch
tercile. Their main effects are therefore perfectly absorbed by region FEs
and must NOT be included as regressors (perfect collinearity). Only the
Post × tercile interactions vary over time and are estimable.

WCB seeds
---------
post_x_medium uses seed=42; post_x_high uses seed=43.
Different seeds ensure different bootstrap weight sequences.
Each test is individually valid. They are not jointly calibrated —
for joint inference use the Wald test (H0: β_M = β_H = 0).

Linearity test
--------------
H0: β_H = 2 × β_M, implemented as a Wald test via PyFixest's wald_test().
Assumes equal tercile spacing. Since the split is 5/4/6 regions (not
equal thirds), the test is approximate — interpret as diagnostic only.
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
    EXPOSURE_TERCILES,
    YEARS,
)

logger = logging.getLogger(__name__)

_PRE_YEARS: list[int] = YEARS   # [2017, 2018, 2019]


# =============================================================================
# BUILD BINNED DiD DATA
# =============================================================================

def build_binned_did_data(
    panel: pl.DataFrame,
    post_years: list[int] | None = None,
) -> pl.DataFrame:
    """
    Prepare the analysis panel for binned DiD estimation.

    Constructs:
      - post           : binary (0 = pre-reform, 1 = post-reform)
      - exposure_tercile: 'low' / 'medium' / 'high' (static from constants)
      - tercile_medium : binary indicator (1 = medium tercile)
      - tercile_high   : binary indicator (1 = high tercile)
      - post_x_medium  : Post × medium (coefficient of interest)
      - post_x_high    : Post × high   (coefficient of interest)

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

    low_regions    = EXPOSURE_TERCILES["low"]
    medium_regions = EXPOSURE_TERCILES["medium"]
    high_regions   = EXPOSURE_TERCILES["high"]

    did = did.with_columns(
        pl.when(pl.col("drgn2").is_in(low_regions))
        .then(pl.lit("low"))
        .when(pl.col("drgn2").is_in(medium_regions))
        .then(pl.lit("medium"))
        .when(pl.col("drgn2").is_in(high_regions))
        .then(pl.lit("high"))
        .otherwise(pl.lit(None))
        .alias("exposure_tercile")
    )

    n_null = did.filter(pl.col("exposure_tercile").is_null()).height
    if n_null > 0:
        unassigned = (
            did.filter(pl.col("exposure_tercile").is_null())
            .select("drgn2").unique().to_series().to_list()
        )
        raise ValueError(
            f"Regions not assigned to any tercile: {unassigned}. "
            f"Update EXPOSURE_TERCILES in constants.py."
        )

    did = did.with_columns(
        pl.col("exposure_tercile").eq("medium").cast(pl.Float64).alias("tercile_medium"),
        pl.col("exposure_tercile").eq("high").cast(pl.Float64).alias("tercile_high"),
    )

    did = did.with_columns(
        (pl.col("post") * pl.col("tercile_medium")).alias("post_x_medium"),
        (pl.col("post") * pl.col("tercile_high")).alias("post_x_high"),
    )

    logger.info(
        "Binned DiD data built: %d obs | pre=%s | post=%s | "
        "low=%d regions, medium=%d, high=%d",
        len(did), pre_years, post_years,
        len(low_regions), len(medium_regions), len(high_regions),
    )
    return did


# =============================================================================
# ESTIMATE ONE OUTCOME
# =============================================================================

def run_binned_did_spec(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
) -> dict:
    """
    Estimate binned DiD for one outcome using PyFixest.

    Parameters
    ----------
    df       : pandas DataFrame (pre-filtered to relevant columns)
    outcome  : outcome column name
    controls : list of control variable column names

    Returns
    -------
    dict with beta_M, beta_H, SEs, CIs, WCB p-values, linearity test
    """
    required = [outcome, "post_x_medium", "post_x_high", "drgn2", "year"] + controls
    df_clean = (
        df[[c for c in required if c in df.columns]]
        .dropna()
        .reset_index(drop=True)
    )

    if len(df_clean) == 0:
        raise ValueError(f"No complete cases for outcome={outcome}")

    # tercile_medium and tercile_high intentionally excluded —
    # they are time-invariant and absorbed by region FEs.
    ctrl_str = (" + " + " + ".join(controls)) if controls else ""
    formula  = f"{outcome} ~ post_x_medium + post_x_high{ctrl_str} | drgn2 + year"

    fit = pf.feols(
        formula,
        data=df_clean,
        vcov={"CRV1": "drgn2"},
    )

    # ── Coefficients ──────────────────────────────────────────────────────────
    coef_M = float(fit.coef()["post_x_medium"])
    se_M   = float(fit.se()["post_x_medium"])
    pval_M = float(fit.pvalue()["post_x_medium"])

    coef_H = float(fit.coef()["post_x_high"])
    se_H   = float(fit.se()["post_x_high"])
    pval_H = float(fit.pvalue()["post_x_high"])

    n_clusters = int(df_clean["drgn2"].nunique())
    t_crit     = float(t_dist.ppf(0.975, df=n_clusters - 1))

    # ── WCB — different seeds for independent bootstrap draws ─────────────────
    p_wbt_M = np.nan
    try:
        boot_M = fit.wildboottest(param="post_x_medium", reps=9999, seed=42)
        raw_M  = boot_M["Pr(>|t|)"]
        p_wbt_M = float(raw_M.iloc[0]) if hasattr(raw_M, "iloc") else float(raw_M)
        logger.info("WCB  -- %s x medium: p = %.4f", outcome, p_wbt_M)
    except Exception as e:
        logger.warning("WCB failed -- %s x medium: %s", outcome, e)

    p_wbt_H = np.nan
    try:
        boot_H = fit.wildboottest(param="post_x_high", reps=9999, seed=43)
        raw_H  = boot_H["Pr(>|t|)"]
        p_wbt_H = float(raw_H.iloc[0]) if hasattr(raw_H, "iloc") else float(raw_H)
        logger.info("WCB  -- %s x high: p = %.4f", outcome, p_wbt_H)
    except Exception as e:
        logger.warning("WCB failed -- %s x high: %s", outcome, e)

    # ── Linearity test H0: beta_H = 2 x beta_M ───────────────────────────────
    lin_f, lin_p, linearity_ratio = np.nan, np.nan, np.nan
    try:
        coef_names = fit.coef().index.tolist()
        idx_M = coef_names.index("post_x_medium")
        idx_H = coef_names.index("post_x_high")
        R = np.zeros((1, len(coef_names)))
        R[0, idx_H] =  1.0
        R[0, idx_M] = -2.0   # H0: beta_H - 2*beta_M = 0
        wald    = fit.wald_test(R=R)
        lin_f   = float(wald["statistic"])
        lin_p   = float(wald["pvalue"])
        if abs(coef_M) > 1e-8:
            linearity_ratio = coef_H / coef_M
    except Exception as e:
        logger.warning("Linearity test failed -- %s: %s", outcome, e)

    logger.info(
        "Binned -- %s: b_M=%+.4f (p_WCB=%.4f) | b_H=%+.4f (p_WCB=%.4f) | "
        "linearity F=%.2f p=%.4f",
        outcome,
        coef_M, p_wbt_M if not np.isnan(p_wbt_M) else -99,
        coef_H, p_wbt_H if not np.isnan(p_wbt_H) else -99,
        lin_f   if not np.isnan(lin_f)   else -99,
        lin_p   if not np.isnan(lin_p)   else -99,
    )

    return {
        "outcome":             outcome,
        "coef_medium":         coef_M,
        "se_medium":           se_M,
        "ci_low_medium":       coef_M - t_crit * se_M,
        "ci_high_medium":      coef_M + t_crit * se_M,
        "pval_cluster_medium": pval_M,
        "pval_wbt_medium":     p_wbt_M,
        "coef_high":           coef_H,
        "se_high":             se_H,
        "ci_low_high":         coef_H - t_crit * se_H,
        "ci_high_high":        coef_H + t_crit * se_H,
        "pval_cluster_high":   pval_H,
        "pval_wbt_high":       p_wbt_H,
        "linearity_ratio":     linearity_ratio,
        "linearity_f":         lin_f,
        "linearity_p":         lin_p,
        "n_obs":               len(df_clean),
        "n_clusters":          n_clusters,
    }


# =============================================================================
# RUN ALL OUTCOMES
# =============================================================================

def run_binned_did(
    did: pl.DataFrame,
    label: str = "baseline",
    outcomes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Estimate binned DiD for all outcomes.

    Parameters
    ----------
    did      : Polars DataFrame from build_binned_did_data()
    label    : label for this estimation window
    outcomes : outcome columns to estimate (default: ANALYSIS_OUTCOMES)

    Returns
    -------
    pd.DataFrame, one row per outcome
    """
    if outcomes is None:
        outcomes = ANALYSIS_OUTCOMES

    df       = did.to_pandas()
    controls = [c for c in BALANCE_CONTROLS if c in df.columns]

    if "post_x_medium" not in df.columns or "post_x_high" not in df.columns:
        raise RuntimeError(
            "Tercile interactions not found. Run build_binned_did_data() first."
        )

    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            logger.warning("Outcome '%s' not in panel -- skipping", outcome)
            continue

        try:
            row = run_binned_did_spec(df, outcome, controls)
            row["label"] = label
            rows.append(row)
        except Exception as e:
            logger.error("Failed -- %s: %s", outcome, e)

    results = pd.DataFrame(rows)
    logger.info(
        "[%s] Binned DiD complete: %d outcomes estimated",
        label, len(results),
    )
    return results
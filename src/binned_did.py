"""
binned_did.py
=============
Binned DiD specification — addresses heterogeneous treatment effects
across the continuous exposure distribution.

Motivation
----------
Callaway, Goodman-Bacon & Sant'Anna (2024) show that TWFE with a
continuous treatment imposes an implicit linear dose-response
assumption. With only 15 clusters and an exposure distribution
spanning −1.02 to +1.90, the linear coefficient may mask:
  - effects concentrated at the top of the exposure distribution
  - near-zero effects in the middle that dilute the average
  - sign reversal when effects are heterogeneous

The binned specification relaxes linearity by estimating separate
ATTs for medium and high exposure terciles relative to low (reference).

Specification
-------------
  Y_hrt = α + β_M (Post_t × 1[medium_r])
              + β_H (Post_t × 1[high_r])
              + γ_r + δ_t + X_hrt θ + ε_hrt

  Reference group: low-exposure regions (País Vasco, Navarra, Asturias,
  Cantabria, Illes Balears) — most generous pre-reform RMI, smallest
  reform-induced change.

Interpretation
--------------
  β_H : average post-reform change in outcome for high-exposure regions
        relative to low-exposure regions
  β_M : same for medium-exposure regions
  β_H ≈ 2 × β_M : linear dose-response supported
  β_H >> β_M    : effects concentrated at top — TWFE dilutes the effect
  Both ≈ 0      : null result is genuine across the distribution

Design decisions
----------------
- Tercile assignment is FIXED in constants.py (not recomputed)
- Low tercile is the reference (clean comparison principle)
- Same WLS + WCB structure as baseline_did.py
- Runs on ANALYSIS_OUTCOMES (matdep, poverty)
- Two post-period definitions: baseline (2021-2025) and COVID-robust (2022-2025)
"""

from __future__ import annotations

import logging
import io
import contextlib

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats

from src.constants import (
    ALL_OUTCOMES,
    BALANCE_CONTROLS,
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
    EXPOSURE_TERCILES,
    REGION_NAMES,
)

logger = logging.getLogger(__name__)

def build_binned_did_data(
    panel: pl.DataFrame,
    post_years: list[int] | None = None,
) -> pl.DataFrame:
    """
    Prepare panel for binned DiD estimation.

    Adds:
      - exposure_tercile : 'low' / 'medium' / 'high' (static, from constants)
      - tercile_medium   : binary indicator
      - tercile_high     : binary indicator
      - post_did         : binary post indicator
      - post_x_medium    : Post × medium
      - post_x_high      : Post × high (low = reference)

    Parameters
    ----------
    panel      : full analysis panel
    post_years : post-reform years (default: DID_POST_YEARS_BASELINE)
    """
    if post_years is None:
        post_years = DID_POST_YEARS_BASELINE

    pre_years  = [2017, 2018, 2019]
    keep_years = pre_years + post_years

    did = panel.filter(pl.col("year").is_in(keep_years))

    # Construct Post indicator
    did = did.with_columns(
        pl.when(pl.col("year").is_in(post_years))
        .then(pl.lit(1.0))
        .when(pl.col("year").is_in(pre_years))
        .then(pl.lit(0.0))
        .otherwise(pl.lit(None))
        .alias("post_did")
    )

    # Assign tercile labels from static dict
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

    # Verify all regions are assigned
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

    # Binary tercile indicators (low is reference, omitted)
    did = did.with_columns(
        pl.col("exposure_tercile").eq("medium").cast(pl.Float64).alias("tercile_medium"),
        pl.col("exposure_tercile").eq("high").cast(pl.Float64).alias("tercile_high"),
    )

    # Post × tercile interactions
    did = did.with_columns(
        (pl.col("post_did") * pl.col("tercile_medium")).alias("post_x_medium"),
        (pl.col("post_did") * pl.col("tercile_high")).alias("post_x_high"),
    )

    logger.info(
        "Binned DiD data built: %d obs | post-years: %s | "
        "low: %d regions, med: %d, high: %d",
        len(did), post_years,
        len(low_regions), len(medium_regions), len(high_regions),
    )
    return did


def run_binned_did(
    did: pl.DataFrame,
    outcome: str,
    controls: list[str] | None = None,
) -> tuple[dict, object, dict[str, float]]:
    """
    Estimate the binned DiD for one outcome.

    Returns
    -------
    result_dict : dict with β_M, β_H, SEs, CIs, WCB p-values
    result      : statsmodels RegressionResultsWrapper
    wbt_results : dict — WCB p-values for post_x_medium and post_x_high
    """
    if controls is None:
        controls = [c for c in BALANCE_CONTROLS if c in did.columns]

    # ── Prepare DataFrame ─────────────────────────────────────────────────────
    keep = (
        ["household_id", "drgn2", "year", outcome, "weight_hh",
         "post_did", "tercile_medium", "tercile_high",
         "post_x_medium", "post_x_high"]
        + controls
    )
    keep = [c for c in keep if c in did.columns]

    df = did.select(keep).to_pandas().dropna(
        subset=[outcome, "post_x_medium", "post_x_high"]
    )
    df = df.reset_index(drop=True)

    # ── Year fixed effects — 2019 reference ───────────────────────────────────
    years_in_sample = sorted(df["year"].unique().tolist())
    ref_year = 2019
    year_dummy_cols = []
    for yr in years_in_sample:
        if yr != ref_year:
            df[f"yr_{yr}"] = (df["year"] == yr).astype(float)
            year_dummy_cols.append(f"yr_{yr}")

    # ── Region FE via dummies ─────────────────────────────────────────────────
    region_dummies = pd.get_dummies(
        df["drgn2"], prefix="reg", drop_first=True
    ).astype(float)
    df = pd.concat([df, region_dummies], axis=1)
    region_cols = region_dummies.columns.tolist()

    ref_code = sorted(df["drgn2"].unique().tolist())[0]
    ref_name = REGION_NAMES.get(int(ref_code), str(ref_code))

    # ── Regressors ────────────────────────────────────────────────────────────
    # post_x_medium and post_x_high are the coefficients of interest
    # Note: tercile_medium and tercile_high are absorbed by region FE
    # (regions don't change tercile over time) — do NOT include them separately
    regressors = (
        ["post_x_medium", "post_x_high"]
        + year_dummy_cols
        + region_cols
        + controls
    )
    regressors = [c for c in regressors if c in df.columns]

    X = sm.add_constant(df[regressors])
    y = df[outcome]
    w = df["weight_hh"]

    # ── Rank check ────────────────────────────────────────────────────────────
    rank   = np.linalg.matrix_rank(X.values)
    n_cols = X.shape[1]
    if rank < n_cols:
        raise ValueError(
            f"Rank deficiency: rank={rank}, n_cols={n_cols}. "
            f"Check for collinearity between tercile indicators and region FE."
        )

    # ── Estimate ──────────────────────────────────────────────────────────────
    model  = sm.WLS(y, X, weights=w)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["drgn2"]},
    )

    # ── Wild cluster bootstrap for both coefficients ──────────────────────────
    wbt_results: dict[str, float] = {}
    try:
        from wildboottest.wildboottest import wildboottest

        for param in ["post_x_medium", "post_x_high"]:
            if param not in result.params.index:
                logger.warning("Skipping WCB for %s — not in model params", param)
                continue

            _trap = io.StringIO()
            with contextlib.redirect_stdout(_trap):
                wbt = wildboottest(
                    model,
                    cluster=df["drgn2"].values,
                    param=param,
                    B=9999,
                    weights_type="webb",
                    seed=42,
                )

            try:
                if hasattr(wbt, "loc"):
                    p_wbt = float(wbt["p-value"].iloc[0])
                elif isinstance(wbt, dict):
                    for key in ["p_value", "pvalue", "Pr(>|t|)", "p-value"]:
                        if key in wbt:
                            raw = wbt[key]
                            p_wbt = float(raw.iloc[0]) if hasattr(raw, "iloc") else float(raw)
                            break
                    else:
                        raise ValueError(f"No p-value key. Keys: {list(wbt.keys())}")
                else:
                    raise ValueError(f"Unexpected type: {type(wbt)}")
                wbt_results[param] = p_wbt
                logger.info(
                    "WCB — %s × %s: p = %.4f", outcome, param, p_wbt
                )
            except Exception as e:
                logger.warning("WCB extraction failed for %s: %s", param, e)

    except ImportError:
        logger.warning("wildboottest not installed.")
    except Exception as e:
        logger.warning("WCB failed: %s", e)

    # ── Extract coefficients ──────────────────────────────────────────────────
    n_clusters = df["drgn2"].nunique()
    t_crit = stats.t.ppf(0.975, df=n_clusters - 1)

    coef_M = result.params.get("post_x_medium", np.nan)
    se_M   = result.bse.get("post_x_medium",   np.nan)
    pval_M = result.pvalues.get("post_x_medium", np.nan)
    p_wbt_M = wbt_results.get("post_x_medium", np.nan)

    coef_H = result.params.get("post_x_high", np.nan)
    se_H   = result.bse.get("post_x_high",   np.nan)
    pval_H = result.pvalues.get("post_x_high", np.nan)
    p_wbt_H = wbt_results.get("post_x_high", np.nan)

    # ── Linearity diagnostic ──────────────────────────────────────────────────
    # If linear dose-response holds: β_H ≈ 2 × β_M
    # Compute ratio for inspection (NOT a formal test)
    if not (np.isnan(coef_M) or np.isnan(coef_H)) and abs(coef_M) > 1e-8:
        linearity_ratio = coef_H / coef_M
    else:
        linearity_ratio = np.nan

    # ── Formal test of equality β_M = β_H/2 (linearity) ───────────────────────
    # H0: β_H = 2 × β_M  →  β_H − 2β_M = 0
    try:
        param_names = result.params.index.tolist()
        R = np.zeros((1, len(param_names)))
        R[0, param_names.index("post_x_high")]   = 1.0
        R[0, param_names.index("post_x_medium")] = -2.0
        lin_test = result.f_test(R)
        lin_f = float(np.squeeze(lin_test.fvalue))
        lin_p = float(lin_test.pvalue)
    except Exception as e:
        logger.warning("Linearity test failed: %s", e)
        lin_f, lin_p = np.nan, np.nan

    result_dict = {
        "outcome":            outcome,
        "coef_medium":        coef_M,
        "se_medium":          se_M,
        "ci_low_medium":      coef_M - t_crit * se_M,
        "ci_high_medium":     coef_M + t_crit * se_M,
        "pval_cluster_medium": pval_M,
        "pval_wbt_medium":    p_wbt_M,
        "coef_high":          coef_H,
        "se_high":            se_H,
        "ci_low_high":        coef_H - t_crit * se_H,
        "ci_high_high":       coef_H + t_crit * se_H,
        "pval_cluster_high":  pval_H,
        "pval_wbt_high":      p_wbt_H,
        "linearity_ratio":    linearity_ratio,
        "linearity_f":        lin_f,
        "linearity_p":        lin_p,
        "n_obs":              int(result.nobs),
        "n_clusters":         n_clusters,
        "r_squared":          result.rsquared,
    }

    logger.info(
        "Binned DiD — %s: β_M=%.4f (p_WCB=%.4f) | β_H=%.4f (p_WCB=%.4f) | "
        "linearity F=%.2f p=%.4f",
        outcome,
        coef_M, p_wbt_M if not np.isnan(p_wbt_M) else -99,
        coef_H, p_wbt_H if not np.isnan(p_wbt_H) else -99,
        lin_f, lin_p,
    )

    return result_dict, result, wbt_results
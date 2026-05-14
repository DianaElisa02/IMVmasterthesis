"""
placebo.py
==========
Placebo reform test for the IMV DiD analysis.

Uses only pre-reform years (2017-2019) and assigns a fake treatment
date at 2019 (Post_fake=1 if year=2019, 0 otherwise, reference=2018).

Purpose
-------
1. Validate parallel trends assumption before the baseline DiD.
2. Specifically address the income pre-trend concern (WCB p=0.070
   for yr_2018_x_exposure in the baseline event study).

If the placebo coefficient β_placebo is zero and insignificant under
wild cluster bootstrap for all three outcomes, the identification
strategy is validated and the income pre-trend is confirmed as noise.

Specification
-------------
  Y_hrt = α + β_placebo (Post_fake_t × Exposure_r)
              + γ_r + δ_t + X_hrt θ + ε_hrt

  where Post_fake_t = 1 if year = 2019 (fake post),
                      0 if year ∈ {2017, 2018} (fake pre)
  and reference year = 2018 (omitted category for year FE)

Design decisions
----------------
- Same region FE, year FE, controls, weights as baseline event study
- Wild cluster bootstrap (Webb, B=9999) for inference
- Cluster at region level (drgn2) — 15 clusters
- All three outcomes estimated: matdep, poverty, income_net_annual
"""

from __future__ import annotations

import logging
import io
import contextlib

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from src.constants import (
    BALANCE_CONTROLS,
    EXPOSURE_SPECS,
    PLACEBO_FAKE_TREATMENT_YEAR,
    PLACEBO_REFERENCE_YEAR,
    PLACEBO_YEARS,
    REGION_NAMES,
)

logger = logging.getLogger(__name__)

PRIMARY_SPEC = EXPOSURE_SPECS[0]


# =============================================================================
# BUILD PLACEBO DATA STRUCTURE
# =============================================================================

def build_placebo_data(panel: pl.DataFrame) -> pl.DataFrame:
    """
    Restrict panel to pre-reform years and construct placebo Post indicator.

    Post_fake = 1 if year == PLACEBO_FAKE_TREATMENT_YEAR (2019)
    Post_fake = 0 if year ∈ {2017, 2018}

    Also constructs the placebo interaction:
      post_fake_x_exposure = Post_fake × exposure_composite_hybrid
    """
    placebo = panel.filter(pl.col("year").is_in(PLACEBO_YEARS))

    placebo = placebo.with_columns(
        pl.when(pl.col("year").eq(PLACEBO_FAKE_TREATMENT_YEAR))
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .alias("post_fake")
    )

    placebo = placebo.with_columns(
        (pl.col("post_fake") * pl.col(PRIMARY_SPEC))
        .alias("post_fake_x_exposure")
    )

    logger.info(
        "Placebo data built: %d obs | years: %s | "
        "fake post = year %d | reference = year %d",
        len(placebo),
        sorted(placebo["year"].unique().to_list()),
        PLACEBO_FAKE_TREATMENT_YEAR,
        PLACEBO_REFERENCE_YEAR,
    )
    return placebo


# =============================================================================
# ESTIMATE PLACEBO
# =============================================================================

def run_placebo(
    placebo: pl.DataFrame,
    outcome: str = "matdep",
    controls: list[str] | None = None,
) -> tuple[pd.DataFrame, object, dict[str, float]]:
    """
    Estimate the placebo DiD specification.

    Parameters
    ----------
    placebo  : pre-reform panel from build_placebo_data()
    outcome  : outcome variable
    controls : household-level controls (defaults to BALANCE_CONTROLS)

    Returns
    -------
    result_table : pd.DataFrame — placebo coefficient, SE, CI, p-values
    result       : statsmodels RegressionResultsWrapper
    wbt_results  : dict — wild bootstrap p-value for placebo interaction
    """
    if controls is None:
        controls = [c for c in BALANCE_CONTROLS if c in placebo.columns]

    # ── Prepare DataFrame ─────────────────────────────────────────────────────
    keep = (
        ["household_id", "drgn2", "year", outcome, "weight_hh",
         "post_fake", "post_fake_x_exposure"]
        + controls
    )
    keep = [c for c in keep if c in placebo.columns]

    df = placebo.select(keep).to_pandas().dropna(
        subset=[outcome, "post_fake_x_exposure"]
    )
    df = df.reset_index(drop=True)
    logger.info(
        "Placebo estimation sample: %d obs | outcome: %s", len(df), outcome
    )

    # ── Year FE: reference year = PLACEBO_REFERENCE_YEAR (2018) ──────────────
    # Only two year dummies possible: yr_2017 and yr_2019
    # yr_2018 omitted as reference category
    for yr in PLACEBO_YEARS:
        if yr != PLACEBO_REFERENCE_YEAR:
            df[f"yr_{yr}"] = (df["year"] == yr).astype(float)

    year_dummy_cols = [f"yr_{yr}" for yr in PLACEBO_YEARS
                       if yr != PLACEBO_REFERENCE_YEAR]

    # ── Region FE via dummies ─────────────────────────────────────────────────
    region_dummies = pd.get_dummies(
        df["drgn2"], prefix="reg", drop_first=True
    ).astype(float)
    df = pd.concat([df, region_dummies], axis=1)
    region_cols = region_dummies.columns.tolist()

    # Log reference region
    ref_code = sorted(df["drgn2"].unique().tolist())[0]
    ref_name = REGION_NAMES.get(int(ref_code), str(ref_code))
    logger.info(
        "Region FE reference: drgn2=%d (%s)", ref_code, ref_name
    )

    # ── Regressors ────────────────────────────────────────────────────────────
    # Placebo interaction (coefficient of interest)
    # Year dummies (year FE — 2018 omitted)
    # Region dummies (region FE)
    # Controls
    regressors = (
        ["post_fake_x_exposure"]
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
            f"Placebo design matrix rank-deficient: rank={rank}, "
            f"n_cols={n_cols}. Check for multicollinearity."
        )
    logger.info(
        "Placebo rank check passed: rank=%d = n_cols=%d ✓", rank, n_cols
    )

    # ── Estimate WLS with cluster-robust SEs ──────────────────────────────────
    model  = sm.WLS(y, X, weights=w)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["drgn2"]},
    )

    logger.info(
        "Placebo estimated — outcome: %s | R²=%.4f | N=%d",
        outcome, result.rsquared, int(result.nobs),
    )

    # ── Wild cluster bootstrap ────────────────────────────────────────────────
    wbt_results: dict[str, float] = {}
    try:
        from wildboottest.wildboottest import wildboottest

        _trap = io.StringIO()
        with contextlib.redirect_stdout(_trap):
            wbt = wildboottest(
                model,
                cluster=df["drgn2"].values,
                param="post_fake_x_exposure",
                B=9999,
                weights_type="webb",
                seed=42,
            )

        logger.debug("Placebo WCB raw output: %s", wbt)

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
                raise ValueError(f"Unexpected return type: {type(wbt)}")

            wbt_results["post_fake_x_exposure"] = p_wbt
            logger.info("Placebo WCB p-value: %.4f", p_wbt)

        except Exception as e:
            logger.warning(
                "Could not extract WCB p-value: %s. Raw: %s", e, wbt
            )

    except ImportError:
        logger.warning("wildboottest not installed.")
    except Exception as e:
        logger.warning("Placebo WCB failed: %s", e)

    # ── Extract placebo coefficient ───────────────────────────────────────────
    # Use t(df=n_clusters-1) critical value for CIs
    n_clusters = df["drgn2"].nunique()
    from scipy import stats
    t_crit = stats.t.ppf(0.975, df=n_clusters - 1)
    logger.info(
        "CI critical value: t(df=%d, 0.975) = %.4f  [clusters=%d]",
        n_clusters - 1, t_crit, n_clusters,
    )

    coef = result.params.get("post_fake_x_exposure", np.nan)
    se   = result.bse.get("post_fake_x_exposure", np.nan)
    pval = result.pvalues.get("post_fake_x_exposure", np.nan)
    p_wbt = wbt_results.get("post_fake_x_exposure", np.nan)

    result_table = pd.DataFrame([{
        "outcome":         outcome,
        "coef":            coef,
        "se":              se,
        "ci_low":          coef - t_crit * se,
        "ci_high":         coef + t_crit * se,
        "pval_cluster":    pval,
        "pval_wbt":        p_wbt,
        "n_obs":           int(result.nobs),
        "n_clusters":      n_clusters,
        "r_squared":       result.rsquared,
        "interpretation":  (
            "PASS — placebo insignificant, parallel trends supported"
            if (not np.isnan(p_wbt) and p_wbt > 0.1)
            else "WARNING — placebo significant, parallel trends concern"
            if not np.isnan(p_wbt)
            else "WCB unavailable — use cluster-robust p-value"
        ),
    }])

    return result_table, result, wbt_results
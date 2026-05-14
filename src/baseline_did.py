"""
baseline_did.py
===============
Baseline DiD estimation for the IMV analysis.

Specification
-------------
  Y_hrt = α + β (Post_t × Exposure_r) + γ_r + δ_t + X_hrt θ + ε_hrt

  where:
    Post_t     = 1 if year ≥ 2021, 0 if year ≤ 2019
    Exposure_r = regional exposure index (one of 5 specs)
    γ_r        = region fixed effects (dummies, Galicia reference)
    δ_t        = year fixed effects (dummies, 2019 reference)
    X_hrt      = household-level controls (BALANCE_CONTROLS)

  Estimated via WLS with survey weights.
  Inference via wild cluster bootstrap (Webb weights, B=9999).
  Standard errors clustered at region level (15 clusters).

Design decisions
----------------
1. Linear probability model for binary outcomes — consistent with FE,
   interpretable as percentage-point changes (Angrist & Pischke 2009)
2. All five exposure specs estimated for robustness
3. Two post-period definitions: full (2021-2025) and COVID-robust (2022-2025)
4. ANALYSIS_OUTCOMES only — income excluded (placebo failure, p=0.081)
5. Wild cluster bootstrap mandatory — 15 clusters too few for asymptotic
   cluster-robust inference (Cameron, Gelbach & Miller 2008)
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
    ANALYSIS_OUTCOMES,
    BALANCE_CONTROLS,
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
    EXPOSURE_SPECS,
    REGION_NAMES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BUILD DiD DATA STRUCTURE
# =============================================================================

def build_did_data(
    panel: pl.DataFrame,
    post_years: list[int] | None = None,
) -> pl.DataFrame:
    """
    Prepare panel for baseline DiD estimation.

    Keeps only pre-reform (post=0) and post-reform (post=1) observations.
    Constructs Post × Exposure interactions for all 5 specs.

    Parameters
    ----------
    panel      : full analysis panel
    post_years : years to include as post-reform (default: DID_POST_YEARS_BASELINE)
    """
    if post_years is None:
        post_years = DID_POST_YEARS_BASELINE

    # Keep pre-reform years and selected post-reform years
    pre_years = [2017, 2018, 2019]
    keep_years = pre_years + post_years

    did = panel.filter(pl.col("year").is_in(keep_years))

    # Construct Post indicator for this window
    did = did.with_columns(
        pl.when(pl.col("year").is_in(post_years))
        .then(pl.lit(1.0))
        .when(pl.col("year").is_in(pre_years))
        .then(pl.lit(0.0))
        .otherwise(pl.lit(None))
        .alias("post_did")
    )

    # Construct Post × Exposure interactions for all specs
    for spec in EXPOSURE_SPECS:
        if spec in did.columns:
            did = did.with_columns(
                (pl.col("post_did") * pl.col(spec))
                .alias(f"post_x_{spec}_did")
            )

    logger.info(
        "DiD data built: %d obs | pre-years: %s | post-years: %s",
        len(did), pre_years, post_years,
    )
    return did


# =============================================================================
# ESTIMATE ONE SPEC
# =============================================================================

def run_did_spec(
    did: pl.DataFrame,
    outcome: str,
    exposure_spec: str,
    controls: list[str] | None = None,
) -> tuple[dict, object, dict[str, float]]:
    """
    Estimate the baseline DiD for one outcome × exposure spec.

    Parameters
    ----------
    did           : panel from build_did_data()
    outcome       : outcome variable
    exposure_spec : one of EXPOSURE_SPECS
    controls      : household-level controls (defaults to BALANCE_CONTROLS)

    Returns
    -------
    result_dict : dict — coefficient, SE, CI, p-values, model stats
    result      : statsmodels RegressionResultsWrapper
    wbt_results : dict — wild bootstrap p-value
    """
    if controls is None:
        controls = [c for c in BALANCE_CONTROLS if c in did.columns]

    interaction_col = f"post_x_{exposure_spec}_did"

    # ── Prepare DataFrame ─────────────────────────────────────────────────────
    keep = (
        ["household_id", "drgn2", "year", outcome, "weight_hh",
         "post_did", interaction_col]
        + controls
    )
    keep = [c for c in keep if c in did.columns]

    df = did.select(keep).to_pandas().dropna(
        subset=[outcome, interaction_col]
    )
    df = df.reset_index(drop=True)

    # ── Year fixed effects — 2019 as reference ────────────────────────────────
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

    # Log reference region
    ref_code = sorted(df["drgn2"].unique().tolist())[0]
    ref_name = REGION_NAMES.get(int(ref_code), str(ref_code))

    # ── Regressors ────────────────────────────────────────────────────────────
    # Post × Exposure (coef of interest) | year dummies | region dummies | controls
    regressors = (
        [interaction_col]
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
            f"Rank deficiency: rank={rank}, n_cols={n_cols}"
        )

    # ── Estimate ──────────────────────────────────────────────────────────────
    model  = sm.WLS(y, X, weights=w)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["drgn2"]},
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
                param=interaction_col,
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
            wbt_results[interaction_col] = p_wbt
            logger.info(
                "WCB — %s × %s: p = %.4f", outcome, exposure_spec, p_wbt
            )
        except Exception as e:
            logger.warning("WCB extraction failed: %s", e)

    except ImportError:
        logger.warning("wildboottest not installed.")
    except Exception as e:
        logger.warning("WCB failed: %s", e)

    # ── Extract coefficient ───────────────────────────────────────────────────
    n_clusters = df["drgn2"].nunique()
    t_crit = stats.t.ppf(0.975, df=n_clusters - 1)

    coef = result.params.get(interaction_col, np.nan)
    se   = result.bse.get(interaction_col, np.nan)
    pval = result.pvalues.get(interaction_col, np.nan)
    p_wbt = wbt_results.get(interaction_col, np.nan)

    result_dict = {
        "outcome":       outcome,
        "exposure_spec": exposure_spec,
        "coef":          coef,
        "se":            se,
        "ci_low":        coef - t_crit * se,
        "ci_high":       coef + t_crit * se,
        "pval_cluster":  pval,
        "pval_wbt":      p_wbt,
        "n_obs":         int(result.nobs),
        "n_clusters":    n_clusters,
        "r_squared":     result.rsquared,
        "ref_region":    ref_name,
    }

    logger.info(
        "DiD — %s × %s: β=%.4f SE=%.4f p_cluster=%.4f p_wbt=%.4f",
        outcome, exposure_spec,
        coef, se, pval,
        p_wbt if not np.isnan(p_wbt) else -99,
    )

    return result_dict, result, wbt_results


# =============================================================================
# RUN ALL SPECS
# =============================================================================

def run_baseline_did(
    did: pl.DataFrame,
    label: str = "baseline",
) -> pd.DataFrame:
    """
    Estimate baseline DiD for all outcomes × all exposure specs.

    Parameters
    ----------
    did   : panel from build_did_data()
    label : label for this estimation (e.g. "baseline" or "covid_robust")

    Returns
    -------
    results_table : pd.DataFrame with one row per outcome × spec
    """
    rows = []
    for outcome in ANALYSIS_OUTCOMES:
        for spec in EXPOSURE_SPECS:
            interaction_col = f"post_x_{spec}_did"
            if interaction_col not in did.columns:
                logger.warning(
                    "Interaction %s not in panel — skipping", interaction_col
                )
                continue
            try:
                result_dict, _, _ = run_did_spec(did, outcome, spec)
                result_dict["label"] = label
                rows.append(result_dict)
            except Exception as e:
                logger.error(
                    "Failed — %s × %s: %s", outcome, spec, e
                )

    return pd.DataFrame(rows)
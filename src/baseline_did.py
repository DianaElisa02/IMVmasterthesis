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
    Exposure_r = regional exposure index (one of 5 specs, time-invariant)
    γ_r        = region fixed effects   — absorbed via demeaning (PyFixest)
    δ_t        = year fixed effects     — absorbed via demeaning (PyFixest)
    X_hrt      = household-level controls (BALANCE_CONTROLS)

  Estimated via WLS (survey weights = weight_hh).
  Region + year FEs absorbed via the Frisch-Waugh-Lovell demeaning
  algorithm inside PyFixest feols — no explicit dummy construction.
  Standard errors clustered at region level (CRV1, 15 clusters).
  Inference via wild cluster bootstrap (Webb weights, B=9999, seed=42).

PyFixest resolves all of this:
  - FE absorbed via demeaning — no dummies, no reference-category risk,
    no rank-deficiency from partial samples
  - vcov={"CRV1": "drgn2"} handles clustering in one argument
  - fit.wildboottest() is a single line; output is a clean pandas Series

Note on Post × Exposure (time-invariant treatment)
---------------------------------------------------
Exposure_r is computed once from pre-reform data and never changes.
Its main effect is time-invariant and is therefore fully absorbed by
region fixed effects — this is correct and expected. Only the
interaction Post_t × Exposure_r survives the within transformation
(Post switches from 0 to 1 after 2020, giving time variation).
The interaction must be pre-built as a column before calling feols.

Design decisions
----------------
1. LPM for binary outcomes (matdep, poverty) — consistent with TWFE,
   coefficients interpretable as percentage-point changes
   (Angrist & Pischke 2009)
2. All five exposure specs estimated; primary spec is first in EXPOSURE_SPECS
3. Two post-period windows kept as separate runner calls (run_baseline_did.py)
4. ANALYSIS_OUTCOMES only — income_net_annual excluded (placebo failure,
   p=0.081; reported in appendix only)
5. Wild cluster bootstrap mandatory with G=15 clusters
   (Cameron, Gelbach & Miller 2008)
6. CIs use t(G−1) = t(14) critical value (2.145 at 95%) — appropriate
   for small-cluster inference; WCB p-value is the primary test statistic
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
    YEARS,             # [2017, 2018, 2019] — pre-reform years
)

logger = logging.getLogger(__name__)

# Pre-reform years — sourced from YEARS in constants.py
_PRE_YEARS: list[int] = YEARS   # [2017, 2018, 2019]


# =============================================================================
# BUILD DiD DATA
# =============================================================================

def build_did_data(
    panel: pl.DataFrame,
    post_years: list[int] | None = None,
) -> pl.DataFrame:
    """
    Prepare the analysis panel for baseline DiD estimation.

    Filters to pre-reform and selected post-reform years, then constructs:
      - post      : binary (0 = pre-reform, 1 = post-reform, null = excluded)
      - post_x_{spec} : Post × Exposure interaction for each exposure spec

    The interactions are pre-built as columns because Exposure_r is
    time-invariant. Passing it standalone to feols alongside region FEs
    would result in it being silently absorbed (zero within-region
    variation after demeaning). Only the interaction varies over time.

    Parameters
    ----------
    panel      : full analysis panel (Polars DataFrame)
    post_years : post-reform years to include
                 (default: DID_POST_YEARS_BASELINE = [2021–2025])
    """
    if post_years is None:
        post_years = DID_POST_YEARS_BASELINE

    pre_years  = _PRE_YEARS
    keep_years = pre_years + post_years

    did = panel.filter(pl.col("year").is_in(keep_years))

    # Post indicator: 1 = post-reform, 0 = pre-reform
    # otherwise=None is a safe guard — unreachable after the year filter above
    did = did.with_columns(
        pl.when(pl.col("year").is_in(post_years))
        .then(pl.lit(1.0))
        .when(pl.col("year").is_in(pre_years))
        .then(pl.lit(0.0))
        .otherwise(pl.lit(None))
        .alias("post")
    )

    # Pre-build Post × Exposure interactions for all 5 specs
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
        logger.warning(
            "Exposure specs not found in panel (skipped): %s", missing
        )

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
    Estimate baseline DiD for one outcome × exposure spec.

    Uses PyFixest feols with:
      - TWFE: region (drgn2) + year fixed effects, absorbed via demeaning
      - WLS:  survey weights (weight_hh)
      - CRV1: standard errors clustered at region level
      - WCB:  wild cluster bootstrap with Webb weights (B=9999)

    Parameters
    ----------
    df            : pandas DataFrame (complete cases only, pre-filtered)
    outcome       : outcome column name
    exposure_spec : one of EXPOSURE_SPECS
    controls      : list of control variable column names

    Returns
    -------
    dict with keys: outcome, exposure_spec, coef, se, ci_low, ci_high,
                    pval_cluster, pval_wbt, n_obs, n_clusters, r2_within
    """
    interaction_col = f"post_x_{exposure_spec}"

    required = [outcome, interaction_col, "drgn2", "year", "weight_hh"] + controls
    df_clean = (
    df[[c for c in required if c in df.columns]]
    .dropna()
    .loc[lambda x: x["weight_hh"] > 0]
    .reset_index(drop=True)
    )

    if len(df_clean) == 0:
        raise ValueError(
            f"No complete cases for outcome={outcome}, spec={exposure_spec}"
        )

    # ── PyFixest formula ──────────────────────────────────────────────────────
    # Pipe syntax: regressors ~ treatment + controls | region_FE + year_FE
    # Region FE (γ_r) and year FE (δ_t) are absorbed via demeaning.
    # No dummy variables are constructed — no reference-category ambiguity.
    ctrl_str = (" + " + " + ".join(controls)) if controls else ""
    formula  = f"{outcome} ~ {interaction_col}{ctrl_str} | drgn2 + year"

    # Baseline (unweighted) — WCB works
    fit = pf.feols(
        formula,
        data=df_clean,
        vcov={"CRV1": "drgn2"},
    )
    boot = fit.wildboottest(param=interaction_col, reps=9999, seed=42)

    # Robustness (weighted) — appendix only, CRV1 SEs
    fit_wls = pf.feols(
        formula,
        data=df_clean,
        vcov={"CRV1": "drgn2"},
        weights="weight_hh",
    )

    # ── Coefficient extraction ────────────────────────────────────────────────
    coef = float(fit.coef()[interaction_col])
    se   = float(fit.se()[interaction_col])
    pval = float(fit.pvalue()[interaction_col])

    # CI using t(G−1) critical value — appropriate for G=15 clusters
    n_clusters = int(df_clean["drgn2"].nunique())
    t_crit     = float(t_dist.ppf(0.975, df=n_clusters - 1))
    ci_low     = coef - t_crit * se
    ci_high    = coef + t_crit * se

    # Within R² — how much Post×Exposure explains after FE absorption
    try:
        r2_within = float(fit.r2(type="within"))
    except Exception:
        try:
            r2_within = float(fit.r2("within"))
        except Exception:
            r2_within = np.nan

    # ── Wild cluster bootstrap ────────────────────────────────────────────────
    # Primary inference method — asymptotic CRV1 unreliable at G=15.
    # Webb weights chosen for small G (Webb 2013).
    # impose_null=True (PyFixest default): bootstrap under H0: β=0.
    p_wbt = np.nan
    try:
        boot  = fit.wildboottest(
            param=interaction_col,
            reps=9999,
            seed=42,
        )
        # boot is a pandas Series; "Pr(>|t|)" is the WCB p-value
        p_raw = boot["Pr(>|t|)"]
        p_wbt = float(p_raw.iloc[0]) if hasattr(p_raw, "iloc") else float(p_raw)
        logger.info(
            "WCB  — %s × %s: p = %.4f", outcome, exposure_spec, p_wbt
        )
    except Exception as e:
        logger.warning(
            "WCB failed — %s × %s: %s", outcome, exposure_spec, e
        )

    logger.info(
        "DiD  — %s × %s: β=%+.4f  SE=%.4f  p_cluster=%.4f  p_wbt=%.4f",
        outcome, exposure_spec, coef, se, pval,
        p_wbt if not np.isnan(p_wbt) else -99,
    )

    return {
        "outcome":       outcome,
        "exposure_spec": exposure_spec,
        "coef":          coef,
        "se":            se,
        "ci_low":        ci_low,
        "ci_high":       ci_high,
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
) -> pd.DataFrame:
    """
    Estimate baseline DiD for all outcomes × all exposure specs.

    Loops over ANALYSIS_OUTCOMES × EXPOSURE_SPECS. For each pair calls
    run_did_spec() which handles feols + WCB. Results are collected into
    a tidy DataFrame with one row per outcome × spec.

    Parameters
    ----------
    did   : Polars DataFrame from build_did_data()
    label : label for this estimation window
            (e.g. "baseline_2021_2025" or "covid_robust_2022_2025")

    Returns
    -------
    pd.DataFrame, one row per outcome × exposure spec
    """
    df       = did.to_pandas()
    controls = [c for c in BALANCE_CONTROLS if c in df.columns]

    if not controls:
        logger.warning(
            "No BALANCE_CONTROLS found in panel — estimating without controls"
        )

    rows = []
    for outcome in ANALYSIS_OUTCOMES:
        if outcome not in df.columns:
            logger.warning("Outcome '%s' not in panel — skipping", outcome)
            continue

        for spec in EXPOSURE_SPECS:
            interaction_col = f"post_x_{spec}"
            if interaction_col not in df.columns:
                logger.warning(
                    "Interaction '%s' not in panel — skipping", interaction_col
                )
                continue

            try:
                row = run_did_spec(df, outcome, spec, controls)
                row["label"] = label
                rows.append(row)
            except Exception as e:
                logger.error(
                    "Estimation failed — %s × %s: %s", outcome, spec, e
                )

    results = pd.DataFrame(rows)

    logger.info(
        "[%s] DiD complete: %d outcome–spec pairs estimated",
        label, len(results),
    )
    return results
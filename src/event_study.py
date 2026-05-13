"""
event_study.py
==============
Event study data structure and estimation for the IMV DiD analysis.

Design decisions
----------------
1. SPECIFICATION
   The event study interacts each year dummy with the continuous exposure
   index (exposure_composite_hybrid). This gives one coefficient per year
   measuring the differential change in outcomes between high- and
   low-exposure regions relative to the reference year (2019).

   Y_hrt = α + Σ_t≠2019 β_t (yr_t × Exposure_r) + γ_r + δ_t + X_hrt θ + ε_hrt

   β_t < 0 for t ∈ pre-reform years → parallel trends violated
   β_t = 0 for t ∈ pre-reform years → parallel trends supported
   β_t < 0 for t ∈ post-reform years → IMV reduced deprivation in
                                        high-exposure regions

2. FIXED EFFECTS
   Region FE (γ_r): absorb time-invariant regional characteristics
   (geography, institutional history, culture). With repeated cross-
   sections, household ID changes each year so entity FE cannot be
   household-level. We implement region FE via region dummies.
   Year FE (δ_t): absorb aggregate time shocks common to all regions
   (business cycle, COVID, inflation). Implemented via year dummies
   already in the specification.

3. STANDARD ERRORS
   Clustered at region level (drgn2) — 15 clusters. With only 15
   clusters, standard cluster-robust SEs are unreliable. We use
   WLS with cluster-robust SEs here for the event study plot, but
   the main inference for the baseline DiD will use wild cluster
   bootstrap (implemented separately in run_baseline_did.py).

4. WEIGHTS
   Survey weights (weight_hh) used throughout via WLS.

5. PRIMARY SPEC ONLY
   Event study uses exposure_composite_hybrid only. Robustness across
   alternative specs is checked in the baseline DiD, not here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from src.constants import (
    BALANCE_CONTROLS,
    EXPOSURE_SPECS,
    EVENT_STUDY_REFERENCE_YEAR,
    EVENT_STUDY_YEARS,
)

logger = logging.getLogger(__name__)

PRIMARY_SPEC = EXPOSURE_SPECS[0]   # exposure_composite_hybrid


# =============================================================================
# STEP 1: BUILD EVENT STUDY DATA STRUCTURE
# =============================================================================

def build_event_study_data(panel: pl.DataFrame) -> pl.DataFrame:
    """
    Add year dummies and year × exposure interactions to the panel.

    For each year in EVENT_STUDY_YEARS (2019 omitted):
      - yr_{year}:              year dummy (1 if obs is from that year)
      - yr_{year}_x_exposure:   year dummy × exposure_composite_hybrid

    The reference year (2019) is omitted — its coefficient is normalised
    to zero by construction. All β_t are interpreted relative to 2019.
    """
    for year in EVENT_STUDY_YEARS:
        panel = panel.with_columns(
            pl.when(pl.col("year").eq(year))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias(f"yr_{year}")
        )
        panel = panel.with_columns(
            (pl.col(f"yr_{year}") * pl.col(PRIMARY_SPEC))
            .alias(f"yr_{year}_x_exposure")
        )

    logger.info(
        "Event study interactions built for years: %s (reference: %s)",
        EVENT_STUDY_YEARS, EVENT_STUDY_REFERENCE_YEAR,
    )
    return panel


# =============================================================================
# STEP 2: ESTIMATE
# =============================================================================

def run_event_study(
    panel: pl.DataFrame,
    outcome: str = "matdep",
    controls: list[str] | None = None,
) -> tuple[pd.DataFrame, object]:
    """
    Estimate the event study via WLS with region and year FE.

    Parameters
    ----------
    panel    : panel with event study columns from build_event_study_data()
    outcome  : outcome variable — "matdep", "poverty", or "income_net_annual"
    controls : household-level controls — defaults to BALANCE_CONTROLS

    Returns
    -------
    coef_table : pd.DataFrame — year, coef, se, ci_low, ci_high
    result     : statsmodels RegressionResultsWrapper — full model output
    """
    if controls is None:
        controls = [c for c in BALANCE_CONTROLS if c in panel.columns]

    interaction_cols = [f"yr_{y}_x_exposure" for y in EVENT_STUDY_YEARS]
    year_dummy_cols  = [f"yr_{y}" for y in EVENT_STUDY_YEARS]

    # ── Prepare pandas DataFrame ──────────────────────────────────────────────
    keep = (
        ["household_id", "drgn2", "year", outcome, "weight_hh"]
        + interaction_cols
        + year_dummy_cols
        + controls
    )
    keep = [c for c in keep if c in panel.columns]

    df = panel.select(keep).to_pandas().dropna(subset=[outcome] + interaction_cols)
    logger.info("Estimation sample: %d observations", len(df))

    # ── Region fixed effects via dummies ──────────────────────────────────────
    # With repeated cross-sections, household ID is unique within year so
    # entity FE cannot absorb region effects. We add region dummies explicitly.
    # drop_first=True drops one region to avoid perfect multicollinearity
    # with the constant.
    region_dummies = pd.get_dummies(
        df["drgn2"], prefix="reg", drop_first=True
    ).astype(float)
    df = pd.concat([df.reset_index(drop=True), region_dummies], axis=1)
    region_cols = region_dummies.columns.tolist()

    # ── Regressors ────────────────────────────────────────────────────────────
    # Year × exposure interactions: these are the coefficients of interest
    # Year dummies: absorb aggregate year shocks (year FE)
    # Region dummies: absorb time-invariant regional characteristics (region FE)
    # Controls: household-level controls to reduce residual variance
    regressors = interaction_cols + year_dummy_cols + region_cols + controls
    regressors = [c for c in regressors if c in df.columns]

    X = sm.add_constant(df[regressors])
    y = df[outcome]
    w = df["weight_hh"]

    # ── Estimate WLS with cluster-robust SEs ──────────────────────────────────
    # Cluster at region level (drgn2). Note: with 15 clusters, these SEs
    # should be treated as indicative only — wild bootstrap is needed for
    # valid inference (implemented in run_baseline_did.py).
    model  = sm.WLS(y, X, weights=w)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["drgn2"]},
    )

    logger.info(
        "Event study estimated — outcome: %s | R²=%.4f | N=%d",
        outcome, result.rsquared, int(result.nobs),
    )

    # ── Extract interaction coefficients ──────────────────────────────────────
    rows = []
    for year in EVENT_STUDY_YEARS:
        col  = f"yr_{year}_x_exposure"
        coef = result.params.get(col, np.nan)
        se   = result.bse.get(col, np.nan)
        rows.append({
            "year":    year,
            "coef":    coef,
            "se":      se,
            "ci_low":  coef - 1.96 * se,
            "ci_high": coef + 1.96 * se,
            "pval":    result.pvalues.get(col, np.nan),
        })

    # Reference year: coefficient normalised to zero by construction
    rows.append({
        "year":    EVENT_STUDY_REFERENCE_YEAR,
        "coef":    0.0,
        "se":      0.0,
        "ci_low":  0.0,
        "ci_high": 0.0,
        "pval":    np.nan,
    })

    coef_table = (
        pd.DataFrame(rows)
        .sort_values("year")
        .reset_index(drop=True)
    )
    return coef_table, result
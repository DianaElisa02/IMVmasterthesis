"""
binned_did.py
=============
Binned DiD specification — relaxes the linear dose-response assumption
of the continuous TWFE by estimating separate ATTs for medium and high
exposure terciles relative to the low tercile (reference group).

Structure
---------
  - _compute_dynamic_terciles() : assigns regions to terciles for a given spec
  - build_binned_did_data()     : constructs post indicator + tercile interactions
  - run_binned_did_spec()       : estimates one outcome with PyFixest + WCB
  - run_binned_did()            : loops over all specs x outcomes, returns tidy DataFrame

Specification
-------------
  Y_hrt = α + β_M (Post_t × 1[medium_r])
              + β_H (Post_t × 1[high_r])
              + γ_r + δ_t + X_hrt·θ + ε_hrt

  Reference group: low-exposure tercile (regions with smallest reform-
  induced change in protection under each given exposure specification).

Tercile assignment
------------------
Terciles are computed dynamically for each exposure specification by
ranking the 17 regions in ascending order of their exposure value and
applying the same 5 / 4 / 6 split used in the primary specification.
For the primary spec (exposure_composite_hybrid) this reproduces the
fixed assignment in EXPOSURE_TERCILES exactly. For alternative specs
the regional ordering changes and terciles are reassigned accordingly.
This makes the binned specification a genuine robustness check: the
poverty signal is tested across different definitions of high exposure.

WCB seeds
---------
Seeds vary by spec index x outcome index to ensure independent bootstrap
draws across all combinations. For spec at index i and outcome at index j:
  seed_M = 42 + i * n_outcomes * 2 + j * 2
  seed_H = seed_M + 1
This produces unique seeds for all 10 combinations (5 specs x 2 outcomes).

Linearity test
--------------
H0: β_H = 2 × β_M, implemented as a Wald test via PyFixest's wald_test().
The test statistic follows a chi-squared distribution (not F) because
the constraint vector q is non-zero (Rβ = q with q != 0). This is
asymptotically valid; with 17 cluster units the approximation is
conservative in the direction of failing to reject linearity.
The ratio b_H/b_M is a descriptive heuristic only and is suppressed
when |b_M| < 0.001 (near-zero denominator).
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
    REGION_NAMES,
    YEARS,
)

logger = logging.getLogger(__name__)

_PRE_YEARS: list[int] = YEARS          # [2017, 2018, 2019]
_N_LOW:     int       = 6              # regions in low tercile
_N_MEDIUM:  int       = 5              # regions in medium tercile

def _compute_dynamic_terciles(
    panel: pl.DataFrame,
    exposure_spec: str,
    n_low: int = _N_LOW,
    n_medium: int = _N_MEDIUM,
) -> dict[str, list[int]]:
    """
    Assign regions to terciles by ranking them on exposure_spec (ascending).

    The bottom n_low regions form the low (reference) tercile, the next
    n_medium form the medium tercile, and the remainder form the high tercile.
    The 5 / 4 / 6 split matches the primary specification.

    For exposure_composite_hybrid this reproduces EXPOSURE_TERCILES exactly.
    For alternative specs the ordering changes and regions are reassigned.

    Parameters
    ----------
    panel         : full analysis panel (Polars DataFrame)
    exposure_spec : name of the exposure column to rank on
    n_low         : number of regions in the low (reference) tercile
    n_medium      : number of regions in the medium tercile

    Returns
    -------
    dict with keys 'low', 'medium', 'high' mapping to lists of drgn2 codes
    """
    if exposure_spec not in panel.columns:
        raise ValueError(
            f"Exposure spec '{exposure_spec}' not found in panel. "
            f"Available specs: {[c for c in panel.columns if 'exposure' in c]}"
        )

    region_exposure = (
        panel
        .select(["drgn2", exposure_spec])
        .unique(subset=["drgn2"])
        .drop_nulls(subset=[exposure_spec]) 
        .sort(exposure_spec, descending=False)
    )

    regions = region_exposure["drgn2"].to_list()
    n_total = len(regions)

    if n_total < n_low + n_medium + 1:
        raise ValueError(
            f"Only {n_total} regions — cannot form three non-empty terciles "
            f"with n_low={n_low}, n_medium={n_medium}."
        )

    return {
        "low":    regions[:n_low],
        "medium": regions[n_low : n_low + n_medium],
        "high":   regions[n_low + n_medium :],
    }



# =============================================================================
# BUILD BINNED DiD DATA
# =============================================================================

def build_binned_did_data(
    panel: pl.DataFrame,
    post_years: list[int] | None = None,
    exposure_spec: str = EXPOSURE_SPECS[0],
) -> pl.DataFrame:
    """
    Prepare the analysis panel for binned DiD estimation.

    Constructs:
      - post             : binary (0 = pre-reform, 1 = post-reform)
      - exposure_tercile : 'low' / 'medium' / 'high' (computed dynamically)
      - tercile_medium   : binary indicator (1 = medium tercile)
      - tercile_high     : binary indicator (1 = high tercile)
      - post_x_medium    : Post x medium (coefficient of interest)
      - post_x_high      : Post x high   (coefficient of interest)

    Tercile boundaries are recomputed for each exposure_spec by ranking
    regions on that spec's values. For the primary spec this reproduces
    the constants.py assignment exactly.

    Parameters
    ----------
    panel         : full analysis panel (Polars DataFrame)
    post_years    : post-reform years (default: DID_POST_YEARS_BASELINE)
    exposure_spec : which exposure specification to use for tercile assignment
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

    terciles       = _compute_dynamic_terciles(did, exposure_spec)
    low_regions    = terciles["low"]
    medium_regions = terciles["medium"]
    high_regions   = terciles["high"]

    low_names    = [REGION_NAMES.get(r, str(r)) for r in low_regions]
    medium_names = [REGION_NAMES.get(r, str(r)) for r in medium_regions]
    high_names   = [REGION_NAMES.get(r, str(r)) for r in high_regions]
    logger.info(
        "  Tercile assignment [%s] | low: %s | medium: %s | high: %s",
        exposure_spec, low_names, medium_names, high_names,
    )

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
        unassigned_names = [REGION_NAMES.get(r, str(r)) for r in unassigned]
        logger.info(
            "  Dropping %d obs from regions with NaN exposure (%s) — "
            "excluded from %s by construction",
            n_null, unassigned_names, exposure_spec,
        )
        did = did.filter(pl.col("exposure_tercile").is_not_null())

    did = did.with_columns(
        pl.col("exposure_tercile").eq("medium").cast(pl.Float64).alias("tercile_medium"),
        pl.col("exposure_tercile").eq("high").cast(pl.Float64).alias("tercile_high"),
    )

    did = did.with_columns(
        (pl.col("post") * pl.col("tercile_medium")).alias("post_x_medium"),
        (pl.col("post") * pl.col("tercile_high")).alias("post_x_high"),
    )

    logger.info(
        "  Data built: %d obs | pre=%s | post=%s | low=%d, medium=%d, high=%d regions",
        len(did), pre_years, post_years,
        len(low_regions), len(medium_regions), len(high_regions),
    )
    return did

def run_binned_did_spec(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    seed_M: int = 42,
    seed_H: int = 43,
) -> dict:
    """
    Estimate binned DiD for one outcome using PyFixest.

    Parameters
    ----------
    df       : pandas DataFrame from build_binned_did_data()
    outcome  : outcome column name
    controls : list of control variable column names
    seed_M   : WCB seed for post_x_medium (varies by spec x outcome)
    seed_H   : WCB seed for post_x_high   (varies by spec x outcome)

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

    ctrl_str = (" + " + " + ".join(controls)) if controls else ""
    formula  = f"{outcome} ~ post_x_medium + post_x_high{ctrl_str} | drgn2 + year"

    fit = pf.feols(
        formula,
        data=df_clean,
        vcov={"CRV1": "drgn2"},
    )

    coef_M = float(fit.coef()["post_x_medium"])
    se_M   = float(fit.se()["post_x_medium"])
    pval_M = float(fit.pvalue()["post_x_medium"])

    coef_H = float(fit.coef()["post_x_high"])
    se_H   = float(fit.se()["post_x_high"])
    pval_H = float(fit.pvalue()["post_x_high"])

    n_clusters = int(df_clean["drgn2"].nunique())
    t_crit     = float(t_dist.ppf(0.975, df=n_clusters - 1))

    p_wbt_M = np.nan
    try:
        boot_M  = fit.wildboottest(param="post_x_medium", reps=9999, seed=seed_M)
        raw_M   = boot_M["Pr(>|t|)"]
        p_wbt_M = float(raw_M.iloc[0]) if hasattr(raw_M, "iloc") else float(raw_M)
        logger.info("    WCB %s x medium: p = %.4f", outcome, p_wbt_M)
    except Exception as e:
        logger.warning("    WCB failed %s x medium: %s", outcome, e)

    p_wbt_H = np.nan
    try:
        boot_H  = fit.wildboottest(param="post_x_high", reps=9999, seed=seed_H)
        raw_H   = boot_H["Pr(>|t|)"]
        p_wbt_H = float(raw_H.iloc[0]) if hasattr(raw_H, "iloc") else float(raw_H)
        logger.info("    WCB %s x high: p = %.4f", outcome, p_wbt_H)
    except Exception as e:
        logger.warning("    WCB failed %s x high: %s", outcome, e)

    lin_stat, lin_p, linearity_ratio = np.nan, np.nan, np.nan
    try:
        coef_names = fit.coef().index.tolist()
        idx_M = coef_names.index("post_x_medium")
        idx_H = coef_names.index("post_x_high")
        R = np.zeros((1, len(coef_names)))
        R[0, idx_H] =  1.0
        R[0, idx_M] = -2.0
        wald     = fit.wald_test(R=R)
        lin_stat = float(wald["statistic"])
        lin_p    = float(wald["pvalue"])
        if abs(coef_M) > 1e-3:
            linearity_ratio = coef_H / coef_M
    except Exception as e:
        logger.warning("    Linearity test failed %s: %s", outcome, e)

    logger.info(
        "    Result %s: b_M=%+.4f (p=%.4f) b_H=%+.4f (p=%.4f) Wald=%.2f p=%.4f",
        outcome,
        coef_M, p_wbt_M if not np.isnan(p_wbt_M) else -99,
        coef_H, p_wbt_H if not np.isnan(p_wbt_H) else -99,
        lin_stat if not np.isnan(lin_stat) else -99,
        lin_p    if not np.isnan(lin_p)    else -99,
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
        "linearity_stat":      lin_stat,
        "linearity_p":         lin_p,
        "n_obs":               len(df_clean),
        "n_clusters":          n_clusters,
    }

def run_binned_did(
    panel: pl.DataFrame,
    post_years: list[int],
    label: str = "baseline",
    outcomes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Estimate binned DiD for all exposure specs x all outcomes.

    For each exposure spec, terciles are recomputed dynamically so that
    the low / medium / high assignment reflects that spec's regional ranking.
    Seeds vary by spec x outcome index for independent bootstrap draws:
      seed_M = 42 + spec_idx * n_outcomes * 2 + outcome_idx * 2
      seed_H = seed_M + 1

    Parameters
    ----------
    panel      : full analysis panel (Polars DataFrame)
    post_years : post-reform years for this estimation window
    label      : label for this estimation window
    outcomes   : outcome columns to estimate (default: ANALYSIS_OUTCOMES)

    Returns
    -------
    pd.DataFrame, one row per exposure spec x outcome
    """
    if outcomes is None:
        outcomes = ANALYSIS_OUTCOMES

    n_outcomes = len(outcomes)
    rows       = []

    for spec_idx, spec in enumerate(EXPOSURE_SPECS):
        logger.info("--- Spec %d/%d: %s ---", spec_idx + 1, len(EXPOSURE_SPECS), spec)

        try:
            did = build_binned_did_data(panel, post_years=post_years, exposure_spec=spec)
        except Exception as e:
            logger.error("Failed to build data for spec=%s: %s", spec, e)
            continue

        df       = did.to_pandas()
        controls = [c for c in BALANCE_CONTROLS if c in df.columns]

        if not controls:
            logger.warning("No BALANCE_CONTROLS found for spec=%s", spec)

        for outcome_idx, outcome in enumerate(outcomes):
            if outcome not in df.columns:
                logger.warning("Outcome '%s' not in panel -- skipping", outcome)
                continue

            seed_M = 42 + spec_idx * n_outcomes * 2 + outcome_idx * 2
            seed_H = seed_M + 1

            try:
                row = run_binned_did_spec(
                    df, outcome, controls, seed_M=seed_M, seed_H=seed_H
                )
                row["label"]         = label
                row["exposure_spec"] = spec
                rows.append(row)
            except Exception as e:
                logger.error("Failed spec=%s outcome=%s: %s", spec, outcome, e)

    results = pd.DataFrame(rows)
    logger.info(
        "[%s] Binned DiD complete: %d spec-outcome pairs estimated",
        label, len(results),
    )
    return results
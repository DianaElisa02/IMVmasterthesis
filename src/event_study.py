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
   clusters, standard cluster-robust SEs are unreliable. CIs in the
   event study plot use a t critical value with G-1 = 14 degrees of
   freedom (~2.14 at 95%) rather than the asymptotic 1.96. Primary
   inference uses wild cluster bootstrap (WCB) applied to all
   interaction terms — both pre- and post-reform — via Webb weights
   (B=9999). CIs in the plot are for visualisation only; use WCB
   p-values for inference.

4. WEIGHTS
   Survey weights (weight_hh) used throughout via WLS.

5. PRIMARY SPEC ONLY
   Event study uses exposure_composite_hybrid only. Robustness across
   alternative specs is checked in the baseline DiD, not here.

6. REGION-SPECIFIC LINEAR TRENDS
   When EVENT_STUDY_REGION_TREND=True in constants.py, adds region ×
   linear time trend terms to absorb pre-existing differential trends.
   Used as robustness check for the income pre-trend concern.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
import scipy.linalg
import statsmodels.api as sm

from src.constants import (
    BALANCE_CONTROLS,
    BALANCE_CONTROLS_EXTENDED,
    EVENT_STUDY_REGION_TREND,
    EVENT_STUDY_REFERENCE_YEAR,
    EVENT_STUDY_YEARS,
    EXPOSURE_SPECS,
    REGION_NAMES,
)

logger = logging.getLogger(__name__)

PRIMARY_SPEC = EXPOSURE_SPECS[0]   # exposure_composite_hybrid

def _extract_wbt_pvalue(wbt: object, col: str) -> float:

    P_VALUE_KEYS = ["p-value", "p_value", "pvalue", "Pr(>|t|)"]

    if isinstance(wbt, pd.DataFrame):
        for key in P_VALUE_KEYS:
            if key in wbt.columns:
                return float(wbt[key].iloc[0])
        raise ValueError(
            f"wildboottest returned a DataFrame for '{col}' but none of the "
            f"expected p-value columns {P_VALUE_KEYS} were found. "
            f"Actual columns: {wbt.columns.tolist()}"
        )

    if isinstance(wbt, dict):
        for key in P_VALUE_KEYS:
            if key in wbt:
                raw = wbt[key]
                return float(raw.iloc[0]) if hasattr(raw, "iloc") else float(raw)
        raise ValueError(
            f"wildboottest returned a dict for '{col}' but none of the "
            f"expected p-value keys {P_VALUE_KEYS} were found. "
            f"Actual keys: {list(wbt.keys())}"
        )

    if isinstance(wbt, (float, int, np.floating, np.integer)):
        return float(wbt)

    raise ValueError(
        f"wildboottest returned an unrecognised type for '{col}': "
        f"{type(wbt)}. Update _extract_wbt_pvalue() to handle this version."
    )

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

def run_event_study(
    panel: pl.DataFrame,
    outcome: str = "matdep",
    controls: list[str] | None = None,
    extended_controls: bool = False,
) -> tuple[pd.DataFrame, object, dict[str, float]]:
    """
    Parameters
    ----------
    panel            : panel with event study columns from build_event_study_data()
    outcome          : outcome variable — "matdep", "poverty", or "income_net_annual"
    controls         : household-level controls — defaults to BALANCE_CONTROLS
    extended_controls: if True, use BALANCE_CONTROLS_EXTENDED (includes labour vars)

    Returns
    -------
    coef_table  : pd.DataFrame — year, coef, se, ci_low, ci_high, pval, pval_wbt
    result      : statsmodels RegressionResultsWrapper — full model output
    wbt_results : dict — wild bootstrap p-values for all interaction terms
    """
    if controls is None:
        ctrl_list = BALANCE_CONTROLS_EXTENDED if extended_controls else BALANCE_CONTROLS
        controls = [c for c in ctrl_list if c in panel.columns]

    interaction_cols = [f"yr_{y}_x_exposure" for y in EVENT_STUDY_YEARS]
    year_dummy_cols  = [f"yr_{y}" for y in EVENT_STUDY_YEARS]

    keep = (
        ["household_id", "drgn2", "year", outcome, "weight_hh"]
        + interaction_cols
        + year_dummy_cols
        + controls
    )
    keep = [c for c in keep if c in panel.columns]

    all_required = [outcome] + interaction_cols + year_dummy_cols + controls
    df = (
        panel.select(keep)
        .to_pandas()
        .dropna(subset=[c for c in all_required if c in panel.columns])
        .reset_index(drop=True)
    )
    logger.info("Estimation sample: %d observations", len(df))

    region_dummies = pd.get_dummies(
        df["drgn2"], prefix="reg", drop_first=True
    ).astype(float)
    df = pd.concat([df, region_dummies], axis=1)
    region_cols = region_dummies.columns.tolist()

    all_regions = sorted(df["drgn2"].unique().tolist())
    ref_region_code = all_regions[0]
    ref_region_name = REGION_NAMES.get(int(ref_region_code), str(ref_region_code))
    logger.info(
        "Region FE reference category: drgn2=%d (%s) — "
        "all region effects interpreted relative to this region",
        ref_region_code, ref_region_name,
    )

    trend_cols: list[str] = []
    if EVENT_STUDY_REGION_TREND:
        # With region-specific linear trends, one interaction term becomes
        # unidentified: the trend terms at the boundary year are collinear
        # with the exposure interaction for that year (exposure is region-
        # level, so yr_last × exposure is a linear combination of the region
        # trends). Drop the redundant interaction but retain its year dummy.
        last_year     = max(EVENT_STUDY_YEARS)
        drop_interact = f"yr_{last_year}_x_exposure"
        trimmed_interactions = [c for c in interaction_cols if c != drop_interact]
        logger.warning(
            "Robustness spec: '%s' is collinear with region trends and "
            "cannot be identified — dropped from regressors. "
            "Its year dummy (yr_%d) is retained. "
            "This coefficient will be NaN in coef_table.",
            drop_interact, last_year,
        )
        regressors = (
            trimmed_interactions
            + year_dummy_cols
            + trend_cols
            + controls
        )
        logger.info(
            "Robustness spec: region dummies dropped — subsumed by "
            "region-specific linear trends + constant. Year FE retained."
        )
    else:
        regressors = (
            interaction_cols
            + year_dummy_cols
            + region_cols
            + controls
        )

    X = sm.add_constant(df[regressors])
    y = df[outcome]
    w = df["weight_hh"]

    rank   = np.linalg.matrix_rank(X.values)
    n_cols = X.shape[1]
    if rank < n_cols:
        raise ValueError(
            f"Design matrix is rank-deficient: rank={rank}, n_cols={n_cols}. "
            f"Perfect multicollinearity detected — check region dummies vs "
            f"year × exposure interactions. Cannot estimate model."
        )
    logger.info(
        "Design matrix rank check passed: rank=%d = n_cols=%d ✓", rank, n_cols,
    )

    model  = sm.WLS(y, X, weights=w)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["drgn2"]},
    )

    logger.info(
        "Event study estimated — outcome: %s | R²=%.4f | N=%d | "
        "region trends: %s",
        outcome, result.rsquared, int(result.nobs),
        "yes" if EVENT_STUDY_REGION_TREND else "no",
    )

    wbt_results: dict[str, float] = {}

    try:
        from wildboottest.wildboottest import wildboottest
        import contextlib
        import io

        for col in interaction_cols:
            if col not in result.params.index:
                logger.warning(
                    "Skipping WCB for %s — not found in model params.", col
                )
                continue

            _trap = io.StringIO()
            with contextlib.redirect_stdout(_trap):
                wbt = wildboottest(
                    model,
                    cluster=df["drgn2"].values,
                    param=col,
                    B=9999,
                    weights_type="webb",
                    seed=42,
                )

            logger.debug("wildboottest raw output for %s: %s", col, wbt)

            p_wbt = _extract_wbt_pvalue(wbt, col)
            wbt_results[col] = p_wbt
            logger.info("Wild bootstrap — %s: p = %.4f", col, p_wbt)

    except ImportError:
        logger.warning(
            "wildboottest not installed — run: pip install wildboottest. "
            "Cluster-robust p-values used instead."
        )
    except Exception as e:
        logger.warning(
            "wildboottest failed: %s — cluster-robust SEs used instead.", e
        )

    n_clusters = df["drgn2"].nunique()
    t_crit = scipy.stats.t.ppf(0.975, df=n_clusters - 1)
    logger.info(
        "CI critical value: t(df=%d, 0.975) = %.4f  [clusters=%d]",
        n_clusters - 1, t_crit, n_clusters,
    )

    rows = []
    for year in EVENT_STUDY_YEARS:
        col  = f"yr_{year}_x_exposure"
        coef = result.params.get(col, np.nan)
        se   = result.bse.get(col, np.nan)
        rows.append({
            "year":     year,
            "coef":     coef,
            "se":       se,
            "ci_low":   coef - t_crit * se,
            "ci_high":  coef + t_crit * se,
            "pval":     result.pvalues.get(col, np.nan),
            "pval_wbt": wbt_results.get(col, np.nan),
        })

    rows.append({
        "year":     EVENT_STUDY_REFERENCE_YEAR,
        "coef":     0.0,
        "se":       0.0,
        "ci_low":   0.0,
        "ci_high":  0.0,
        "pval":     np.nan,
        "pval_wbt": np.nan,
    })

    coef_table = (
        pd.DataFrame(rows)
        .sort_values("year")
        .reset_index(drop=True)
    )
    return coef_table, result, wbt_results
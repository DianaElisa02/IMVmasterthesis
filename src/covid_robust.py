"""
covid_robust.py
===============
COVID robustness check for the IMV DiD event study.

Motivation
----------
The baseline event study shows positive and significant post-reform
coefficients for matdep in 2021-2022 under wild cluster bootstrap,
suggesting high-exposure regions worsened relative to low-exposure
regions post-reform. The most likely explanation is COVID confounding:
southern high-exposure regions (Andalucía, Valencia, Canarias, Murcia)
were hit harder by the COVID economic shock in 2020-2022 than northern
low-exposure regions (País Vasco, Navarra, Asturias).

Two robustness specs are estimated:
  - excl_2021:      excludes 2021 only (peak COVID + incomplete IMV rollout)
  - excl_2021_2022: excludes 2021 and 2022 (COVID recovery + low take-up)

If positive matdep coefficients shrink, turn insignificant, or reverse
sign when early post-reform years are excluded, COVID confounding is
confirmed as the primary driver of the baseline result.

Specification
-------------
Identical to baseline event study but with excluded years removed:

  Y_hrt = α + Σ_t≠2019 β_t (yr_t × Exposure_r) + γ_r + δ_t + X θ + ε

Design decisions
----------------
- Same region FE, year FE, controls, weights as baseline event study
- Wild cluster bootstrap (Webb, B=9999) for inference
- Runs on ANALYSIS_OUTCOMES only (matdep, poverty)
- income_net_annual excluded — parallel trends violated (placebo p=0.081)
- Reference year: 2019 (unchanged)
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
    COVID_ROBUST_SPECS,
    EVENT_STUDY_REFERENCE_YEAR,
    EXPOSURE_SPECS,
    REGION_NAMES,
)

logger = logging.getLogger(__name__)

PRIMARY_SPEC = EXPOSURE_SPECS[0]

def build_covid_robust_data(
    panel: pl.DataFrame,
    spec_name: str = "excl_2021",
) -> tuple[pl.DataFrame, dict]:
    """
    Restrict panel and build year × exposure interactions for a given
    COVID robustness spec.

    Parameters
    ----------
    panel     : full analysis panel
    spec_name : key in COVID_ROBUST_SPECS — "excl_2021" or "excl_2021_2022"

    Returns
    -------
    robust : filtered panel with year × exposure interactions rebuilt
    spec   : the spec dict for use downstream
    """
    spec              = COVID_ROBUST_SPECS[spec_name]
    exclude_years     = spec["exclude_years"]
    event_study_years = spec["event_study_years"]

    robust = panel.filter(~pl.col("year").is_in(exclude_years))

    n_dropped = len(panel) - len(robust)
    logger.info(
        "COVID robust [%s]: dropped %d obs for years %s | remaining: %d obs",
        spec_name, n_dropped, exclude_years, len(robust),
    )

    for year in event_study_years:
        robust = robust.with_columns(
            pl.when(pl.col("year").eq(year))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias(f"yr_{year}")
        )
        robust = robust.with_columns(
            (pl.col(f"yr_{year}") * pl.col(PRIMARY_SPEC))
            .alias(f"yr_{year}_x_exposure")
        )

    logger.info(
        "COVID robust [%s] interactions built for years: %s (reference: %s)",
        spec_name, event_study_years, EVENT_STUDY_REFERENCE_YEAR,
    )
    return robust, spec


def run_covid_robust(
    robust: pl.DataFrame,
    spec: dict,
    outcome: str = "matdep",
    controls: list[str] | None = None,
) -> tuple[pd.DataFrame, object, dict[str, float]]:
    """
    Estimate the COVID robustness event study for a given spec.

    Parameters
    ----------
    robust   : panel from build_covid_robust_data()
    spec     : spec dict from COVID_ROBUST_SPECS
    outcome  : outcome variable
    controls : household-level controls (defaults to BALANCE_CONTROLS)

    Returns
    -------
    coef_table  : pd.DataFrame — year, coef, se, ci_low, ci_high, pval, pval_wbt
    result      : statsmodels RegressionResultsWrapper
    wbt_results : dict — wild bootstrap p-values for all interactions
    """
    if controls is None:
        controls = [c for c in BALANCE_CONTROLS if c in robust.columns]

    event_study_years = spec["event_study_years"]
    interaction_cols  = [f"yr_{y}_x_exposure" for y in event_study_years]
    year_dummy_cols   = [f"yr_{y}" for y in event_study_years]

    keep = (
        ["household_id", "drgn2", "year", outcome, "weight_hh"]
        + interaction_cols
        + year_dummy_cols
        + controls
    )
    keep = [c for c in keep if c in robust.columns]

    df = robust.select(keep).to_pandas().dropna(
        subset=[outcome] + interaction_cols
    )
    df = df.reset_index(drop=True)
    logger.info(
        "COVID robust estimation sample: %d obs | outcome: %s",
        len(df), outcome,
    )

    region_dummies = pd.get_dummies(
        df["drgn2"], prefix="reg", drop_first=True
    ).astype(float)
    df = pd.concat([df, region_dummies], axis=1)
    region_cols = region_dummies.columns.tolist()

    ref_code = sorted(df["drgn2"].unique().tolist())[0]
    ref_name = REGION_NAMES.get(int(ref_code), str(ref_code))
    logger.info(
        "Region FE reference: drgn2=%d (%s)", ref_code, ref_name
    )

    regressors = (
        interaction_cols
        + year_dummy_cols
        + region_cols
        + controls
    )
    regressors = [c for c in regressors if c in df.columns]

    X = sm.add_constant(df[regressors])
    y = df[outcome]
    w = df["weight_hh"]

    rank   = np.linalg.matrix_rank(X.values)
    n_cols = X.shape[1]
    if rank < n_cols:
        raise ValueError(
            f"Rank deficiency: rank={rank}, n_cols={n_cols}. "
            f"Check for multicollinearity."
        )
    logger.info(
        "Rank check passed: rank=%d = n_cols=%d ✓", rank, n_cols
    )

    model  = sm.WLS(y, X, weights=w)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["drgn2"]},
    )

    logger.info(
        "COVID robust estimated — outcome: %s | R²=%.4f | N=%d | spec: %s",
        outcome, result.rsquared, int(result.nobs), spec["label"],
    )

    wbt_results: dict[str, float] = {}
    try:
        from wildboottest.wildboottest import wildboottest

        for col in interaction_cols:
            if col not in result.params.index:
                logger.warning("Skipping WCB for %s — not in model params", col)
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

            logger.debug("WCB raw output for %s: %s", col, wbt)

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

                wbt_results[col] = p_wbt
                logger.info("WCB — %s: p = %.4f", col, p_wbt)

            except Exception as e:
                logger.warning(
                    "WCB extraction failed for %s: %s. Raw: %s", col, e, wbt
                )

    except ImportError:
        logger.warning("wildboottest not installed.")
    except Exception as e:
        logger.warning("WCB failed: %s", e)

    n_clusters = df["drgn2"].nunique()
    t_crit = stats.t.ppf(0.975, df=n_clusters - 1)
    logger.info(
        "CI critical value: t(df=%d, 0.975) = %.4f  [clusters=%d]",
        n_clusters - 1, t_crit, n_clusters,
    )

    rows = []
    for year in event_study_years:
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
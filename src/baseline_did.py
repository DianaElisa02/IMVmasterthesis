"""
baseline_did.py
===============
Continuous DiD estimation for the IMV analysis.

This module estimates the continuous-exposure DiD specifications used as
dose-response benchmarks. The primary baseline specification is the binned
/ tercile DiD estimated in binned_did.py.

Specification
-------------
  Y_hrt = α + β (Post_t × Exposure_r) + γ_r + δ_t + X_hrt·θ + ε_hrt

where:
  Post_t     = 1 if year in post_years, 0 if year in [2017, 2018, 2019]
  Exposure_r = regional exposure index, time-invariant
  γ_r        = region fixed effects, absorbed by PyFixest
  δ_t        = year fixed effects, absorbed by PyFixest
  X_hrt      = household-level controls

Estimated by unweighted OLS with region and year fixed effects.
Standard errors are clustered at the region level.
Inference uses wild cluster bootstrap with 9,999 replications.

Important implementation note
-----------------------------
Categorical controls are cast to pandas 'category' before estimation and are
included in the formula as plain column names. Do not wrap them in i(...),
because PyFixest's wildboottest may fail when re-evaluating formulas
containing i().
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

_PRE_YEARS: list[int] = YEARS  # [2017, 2018, 2019]

# These controls are substantively categorical. They are cast to pandas
# category before estimation. Do not wrap them in i(...), because WCB can fail
# when re-evaluating formulas containing i().
CATEGORICAL_CONTROLS = {
    "head_age_group",
    "head_sex",
    "head_labour_group",
}


def _control_terms_for_formula(controls: list[str]) -> list[str]:
    """
    Return formula terms for controls.

    Categorical controls are kept as plain column names because they are cast
    to pandas 'category' before estimation.
    """
    return controls


def _prepare_controls_for_formula(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure categorical controls are treated as factors by the formula parser.
    """
    out = df.copy()

    for col in CATEGORICAL_CONTROLS:
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out


def _extract_boot_pvalue(boot) -> float:
    """Extract p-value from PyFixest wildboottest output across versions."""
    for key in ["Pr(>|t|)", "p-value", "pvalue", "p_value"]:
        try:
            if key in boot:
                value = boot[key]
                if hasattr(value, "iloc"):
                    return float(value.iloc[0])
                return float(value)
        except Exception:
            continue
    return np.nan


def _run_wcb(fit, param: str, seed: int, reps: int = 9999) -> float:
    """Run wild cluster bootstrap and return p-value."""
    try:
        boot = fit.wildboottest(param=param, reps=reps, seed=seed)
        return _extract_boot_pvalue(boot)
    except Exception as exc:
        logger.warning("WCB failed for %s: %s", param, exc)
        return np.nan


# =============================================================================
# BUILD DiD DATA
# =============================================================================
def build_did_data(
    panel: pl.DataFrame,
    post_years: list[int] | None = None,
) -> pl.DataFrame:
    """
    Prepare the analysis panel for continuous DiD estimation.

    Constructs:
      - post          : binary, 0 in pre-reform years and 1 in post-reform years
      - post_x_{spec} : post x exposure interaction for each exposure spec
    """
    if post_years is None:
        post_years = DID_POST_YEARS_BASELINE

    pre_years = _PRE_YEARS
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
        len(did),
        pre_years,
        post_years,
        built,
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
    seed: int = 42,
) -> dict:
    """
    Estimate one continuous DiD specification.
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

    df_clean = _prepare_controls_for_formula(df_clean)

    control_terms = _control_terms_for_formula(controls)
    ctrl_str = (" + " + " + ".join(control_terms)) if control_terms else ""
    formula = f"{outcome} ~ {interaction_col}{ctrl_str} | drgn2 + year"

    fit = pf.feols(
        formula,
        data=df_clean,
        vcov={"CRV1": "drgn2"},
    )

    coef = float(fit.coef()[interaction_col])
    se = float(fit.se()[interaction_col])
    pval = float(fit.pvalue()[interaction_col])

    n_clusters = int(df_clean["drgn2"].nunique())
    t_crit = float(t_dist.ppf(0.975, df=n_clusters - 1))

    # Within R²
    r2_within = np.nan
    try:
        r2_within = float(fit.r2(type="within"))
    except Exception:
        try:
            r2_within = float(fit.r2("within"))
        except Exception:
            logger.warning(
                "R² within extraction failed -- %s x %s: "
                "r2_within set to NaN in results CSV",
                outcome,
                exposure_spec,
            )

    p_wbt = _run_wcb(fit, interaction_col, seed=seed, reps=9999)

    logger.info(
        "DiD -- %s x %s: b=%+.4f SE=%.4f p_cluster=%.4f p_wbt=%.4f",
        outcome,
        exposure_spec,
        coef,
        se,
        pval,
        p_wbt if not np.isnan(p_wbt) else -99,
    )

    return {
        "outcome": outcome,
        "exposure_spec": exposure_spec,
        "coef": coef,
        "se": se,
        "ci_low": coef - t_crit * se,
        "ci_high": coef + t_crit * se,
        "pval_cluster": pval,
        "pval_wbt": p_wbt,
        "n_obs": len(df_clean),
        "n_clusters": n_clusters,
        "r2_within": r2_within,
    }


def run_baseline_did(
    did: pl.DataFrame,
    label: str = "baseline",
    outcomes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Estimate continuous DiD for all outcomes x all exposure specs.
    """
    if outcomes is None:
        outcomes = ANALYSIS_OUTCOMES

    df = did.to_pandas()
    controls = [c for c in BALANCE_CONTROLS if c in df.columns]

    if not controls:
        logger.warning("No BALANCE_CONTROLS found -- estimating without controls")

    rows = []

    for outcome_idx, outcome in enumerate(outcomes):
        if outcome not in df.columns:
            logger.warning("Outcome '%s' not in panel -- skipping", outcome)
            continue

        for spec_idx, spec in enumerate(EXPOSURE_SPECS):
            interaction_col = f"post_x_{spec}"
            if interaction_col not in df.columns:
                logger.warning(
                    "Interaction '%s' not in panel -- skipping",
                    interaction_col,
                )
                continue

            seed = 42 + outcome_idx * len(EXPOSURE_SPECS) + spec_idx

            try:
                row = run_did_spec(
                    df=df,
                    outcome=outcome,
                    exposure_spec=spec,
                    controls=controls,
                    seed=seed,
                )
                row["label"] = label
                rows.append(row)
            except Exception as exc:
                logger.error("Failed -- %s x %s: %s", outcome, spec, exc)

    results = pd.DataFrame(rows)

    logger.info(
        "[%s] DiD complete: %d outcome-spec pairs estimated",
        label,
        len(results),
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

    Uses only pre-reform years, 2017--2019.
    Fake treatment: 2019.
    """
    if exposure_spec is None:
        exposure_spec = EXPOSURE_SPECS[0]

    if exposure_spec not in panel.columns:
        raise ValueError(f"Exposure spec '{exposure_spec}' not found in panel.")

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
    else:
        controls = [c for c in controls if c in df.columns]

    control_terms = _control_terms_for_formula(controls)
    ctrl_str = (" + " + " + ".join(control_terms)) if control_terms else ""

    rows = []

    for outcome_idx, outcome in enumerate(outcomes):
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

        df_clean = _prepare_controls_for_formula(df_clean)

        formula = f"{outcome} ~ post_fake_x_exposure{ctrl_str} | drgn2 + year"

        fit = pf.feols(formula, data=df_clean, vcov={"CRV1": "drgn2"})

        coef = float(fit.coef()["post_fake_x_exposure"])
        se = float(fit.se()["post_fake_x_exposure"])
        pval = float(fit.pvalue()["post_fake_x_exposure"])

        n_clusters = int(df_clean["drgn2"].nunique())
        t_crit = float(t_dist.ppf(0.975, df=n_clusters - 1))

        seed = 42 + outcome_idx
        p_wbt = _run_wcb(fit, "post_fake_x_exposure", seed=seed, reps=9999)

        verdict = (
            "PASS"
            if not np.isnan(p_wbt) and p_wbt > 0.10
            else "WARNING"
        )

        logger.info(
            "Placebo -- %s: b=%+.4f SE=%.4f p_cluster=%.4f p_wbt=%.4f -> %s",
            outcome,
            coef,
            se,
            pval,
            p_wbt if not np.isnan(p_wbt) else -99,
            verdict,
        )

        rows.append({
            "outcome": outcome,
            "exposure_spec": exposure_spec,
            "coef": coef,
            "se": se,
            "ci_low": coef - t_crit * se,
            "ci_high": coef + t_crit * se,
            "pval_cluster": pval,
            "pval_wbt": p_wbt,
            "n_obs": len(df_clean),
            "n_clusters": n_clusters,
            "verdict": verdict,
        })

    results = pd.DataFrame(rows)

    if not results.empty:
        passed = results[results["verdict"] == "PASS"]["outcome"].tolist()
        warned = results[results["verdict"] == "WARNING"]["outcome"].tolist()

        if passed:
            logger.info("Placebo PASSED: %s", passed)
        if warned:
            logger.warning("Placebo WARNING: %s", warned)

    return results
"""Raw-unit and joint continuous DiD extensions."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
from scipy.stats import t as t_dist

from src.constants import ANALYSIS_OUTCOMES, YEARS
from src.control_specs import PREFERRED_CONTROLS, add_preferred_control_groups, cast_categorical_controls
from src.exposure_specs import PRIMARY_EXPOSURE_SPECS, RAW_EXPOSURE_MAP

logger = logging.getLogger(__name__)


def attach_raw_exposures(panel: pl.DataFrame, exposure_csv: Path) -> pl.DataFrame:
    raw_cols = list(RAW_EXPOSURE_MAP.values())
    missing = [col for col in raw_cols if col not in panel.columns]
    if not missing:
        return panel
    exposure = pl.read_csv(exposure_csv)
    available = [col for col in missing if col in exposure.columns]
    if not available:
        raise ValueError(f"No raw exposure columns found in {exposure_csv}")
    regional = exposure.select([pl.col("drgn2").cast(pl.Int32)] + available)
    if regional["drgn2"].n_unique() != regional.height:
        raise ValueError("Duplicate regional rows in exposure CSV")
    return panel.join(regional, on="drgn2", how="left")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    return cast_categorical_controls(df)


def _boot_pvalue(boot) -> float:
    for key in ["Pr(>|t|)", "p-value", "pvalue", "p_value"]:
        try:
            value = boot[key]
            return float(value.iloc[0] if hasattr(value, "iloc") else value)
        except Exception:
            pass
    return np.nan


def _preferred_controls(df: pd.DataFrame) -> list[str]:
    missing = [control for control in PREFERRED_CONTROLS if control not in df.columns]
    if missing:
        raise ValueError(f"Preferred controls missing from analysis data: {missing}")
    return PREFERRED_CONTROLS


def run_raw_continuous_models(panel: pl.DataFrame, post_years: list[int], label: str, outcomes: list[str] | None = None) -> pd.DataFrame:
    outcomes = outcomes or ANALYSIS_OUTCOMES
    panel = add_preferred_control_groups(panel)
    raw_specs = [RAW_EXPOSURE_MAP[spec] for spec in PRIMARY_EXPOSURE_SPECS]
    did = panel.filter(pl.col("year").is_in(YEARS + post_years)).with_columns(
        pl.col("year").is_in(post_years).cast(pl.Float64).alias("post")
    )
    for raw in raw_specs:
        if raw not in did.columns:
            raise ValueError(f"Raw exposure '{raw}' missing")
        did = did.with_columns((pl.col("post") * pl.col(raw)).alias(f"post_x_{raw}"))
    df = did.to_pandas()
    controls = _preferred_controls(df)
    rows = []
    for outcome_idx, outcome in enumerate(outcomes):
        for spec_idx, raw in enumerate(raw_specs):
            term = f"post_x_{raw}"
            required = [outcome, term, "drgn2", "year"] + controls
            clean = _prepare(df[required].dropna())
            ctrl = (" + " + " + ".join(controls)) if controls else ""
            fit = pf.feols(f"{outcome} ~ {term}{ctrl} | drgn2 + year", data=clean, vcov={"CRV1": "drgn2"})
            g = int(clean["drgn2"].nunique())
            crit = float(t_dist.ppf(0.975, df=g - 1))
            coef, se = float(fit.coef()[term]), float(fit.se()[term])
            try:
                p_wbt = _boot_pvalue(fit.wildboottest(param=term, reps=9999, seed=500 + outcome_idx * 10 + spec_idx))
            except Exception:
                p_wbt = np.nan
            rows.append({
                "label": label,
                "scale": "raw",
                "outcome": outcome,
                "exposure_spec": raw,
                "coef": coef,
                "se": se,
                "ci_low": coef - crit * se,
                "ci_high": coef + crit * se,
                "pval_cluster": float(fit.pvalue()[term]),
                "pval_wbt": p_wbt,
                "n_obs": len(clean),
                "n_clusters": g,
                "controls": ";".join(controls),
                "control_spec": "preferred_demographic",
            })
    return pd.DataFrame(rows)


def run_joint_standardised_model(panel: pl.DataFrame, post_years: list[int], label: str, outcomes: list[str] | None = None) -> pd.DataFrame:
    outcomes = outcomes or ANALYSIS_OUTCOMES
    panel = add_preferred_control_groups(panel)
    did = panel.filter(pl.col("year").is_in(YEARS + post_years)).with_columns(
        pl.col("year").is_in(post_years).cast(pl.Float64).alias("post")
    )
    terms = []
    for spec in PRIMARY_EXPOSURE_SPECS:
        term = f"post_x_{spec}"
        did = did.with_columns((pl.col("post") * pl.col(spec)).alias(term))
        terms.append(term)
    df = did.to_pandas()
    controls = _preferred_controls(df)
    rows = []
    for outcome_idx, outcome in enumerate(outcomes):
        required = [outcome, "drgn2", "year"] + terms + controls
        clean = _prepare(df[required].dropna())
        ctrl = (" + " + " + ".join(controls)) if controls else ""
        fit = pf.feols(f"{outcome} ~ {' + '.join(terms)}{ctrl} | drgn2 + year", data=clean, vcov={"CRV1": "drgn2"})
        g = int(clean["drgn2"].nunique())
        crit = float(t_dist.ppf(0.975, df=g - 1))
        for spec_idx, (spec, term) in enumerate(zip(PRIMARY_EXPOSURE_SPECS, terms)):
            coef, se = float(fit.coef()[term]), float(fit.se()[term])
            try:
                p_wbt = _boot_pvalue(fit.wildboottest(param=term, reps=9999, seed=800 + outcome_idx * 10 + spec_idx))
            except Exception:
                p_wbt = np.nan
            rows.append({
                "label": label,
                "scale": "standardised_joint",
                "outcome": outcome,
                "exposure_spec": spec,
                "coef": coef,
                "se": se,
                "ci_low": coef - crit * se,
                "ci_high": coef + crit * se,
                "pval_cluster": float(fit.pvalue()[term]),
                "pval_wbt": p_wbt,
                "n_obs": len(clean),
                "n_clusters": g,
                "controls": ";".join(controls),
                "control_spec": "preferred_demographic",
            })
    return pd.DataFrame(rows)

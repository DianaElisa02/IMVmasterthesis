"""Baseline tercile DiD for the IMV analysis.

Regions are ranked separately for every exposure measure. Low exposure is the
comparison group; medium and high exposure effects are estimated relative to it.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
from scipy.stats import t as t_dist

from src.constants import ANALYSIS_OUTCOMES, BALANCE_CONTROLS, DID_POST_YEARS_BASELINE, EXPOSURE_SPECS, REGION_NAMES, YEARS

logger = logging.getLogger(__name__)
_PRE_YEARS = YEARS
_CATEGORICAL_CONTROLS = {
    "head_age_group",
    "head_sex",
    "head_labour_group",
    "head_education_group",
    "n_adults_group",
    "n_children_group",
}


def _balanced_group_sizes(n: int) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError("At least three regions are required for tercile estimation")
    base, rem = divmod(n, 3)
    if rem == 0:
        return base, base, base
    if rem == 1:
        return base + 1, base, base
    return base + 1, base, base + 1


def compute_tercile_assignments(panel: pl.DataFrame, exposure_spec: str) -> pd.DataFrame:
    if exposure_spec not in panel.columns:
        raise ValueError(f"Exposure '{exposure_spec}' not found")
    regional = (
        panel.select(["drgn2", exposure_spec])
        .group_by("drgn2")
        .agg(
            pl.col(exposure_spec).drop_nulls().n_unique().alias("n_values"),
            pl.col(exposure_spec).drop_nulls().first().alias("exposure_value"),
        )
        .filter(pl.col("n_values") > 0)
    )
    bad = regional.filter(pl.col("n_values") > 1)
    if bad.height:
        raise ValueError(f"Exposure varies within region: {bad['drgn2'].to_list()}")
    regional = regional.sort(["exposure_value", "drgn2"])
    n_low, n_medium, _ = _balanced_group_sizes(regional.height)
    pdf = regional.select(["drgn2", "exposure_value"]).to_pandas()
    pdf["rank"] = np.arange(1, len(pdf) + 1)
    pdf["exposure_tercile"] = np.where(
        pdf["rank"] <= n_low,
        "low",
        np.where(pdf["rank"] <= n_low + n_medium, "medium", "high"),
    )
    pdf["region_name"] = pdf["drgn2"].map(REGION_NAMES).fillna(pdf["drgn2"].astype(str))
    pdf["exposure_spec"] = exposure_spec
    ties = pdf.groupby("exposure_value")["drgn2"].count()
    if (ties > 1).any():
        logger.warning("Ties in %s resolved deterministically by drgn2", exposure_spec)
    return pdf


def build_binned_did_data(panel: pl.DataFrame, post_years: list[int] | None = None, exposure_spec: str = EXPOSURE_SPECS[0]) -> tuple[pl.DataFrame, pd.DataFrame]:
    post_years = post_years or DID_POST_YEARS_BASELINE
    did = panel.filter(pl.col("year").is_in(_PRE_YEARS + post_years)).with_columns(
        pl.col("year").is_in(post_years).cast(pl.Float64).alias("post")
    )
    assignments = compute_tercile_assignments(did, exposure_spec)
    assignment_pl = pl.from_pandas(assignments[["drgn2", "exposure_tercile"]])
    did = did.join(assignment_pl, on="drgn2", how="inner").with_columns(
        pl.col("exposure_tercile").eq("medium").cast(pl.Float64).alias("tercile_medium"),
        pl.col("exposure_tercile").eq("high").cast(pl.Float64).alias("tercile_high"),
    ).with_columns(
        (pl.col("post") * pl.col("tercile_medium")).alias("post_x_medium"),
        (pl.col("post") * pl.col("tercile_high")).alias("post_x_high"),
    )
    logger.info("Groups [%s]: %s", exposure_spec, assignments.groupby("exposure_tercile")["region_name"].apply(list).to_dict())
    return did, assignments


def _boot_pvalue(boot) -> float:
    for key in ["Pr(>|t|)", "p-value", "pvalue", "p_value"]:
        try:
            value = boot[key]
            return float(value.iloc[0] if hasattr(value, "iloc") else value)
        except Exception:
            pass
    return np.nan


def run_binned_did_spec(df: pd.DataFrame, outcome: str, controls: list[str], seed_medium: int, seed_high: int) -> dict:
    required = [outcome, "post_x_medium", "post_x_high", "drgn2", "year"] + controls
    clean = df[required].dropna().reset_index(drop=True)
    for col in _CATEGORICAL_CONTROLS:
        if col in clean.columns:
            clean[col] = clean[col].astype("category")
    ctrl = (" + " + " + ".join(controls)) if controls else ""
    fit = pf.feols(f"{outcome} ~ post_x_medium + post_x_high{ctrl} | drgn2 + year", data=clean, vcov={"CRV1": "drgn2"})
    g = int(clean["drgn2"].nunique())
    crit = float(t_dist.ppf(0.975, df=g - 1))
    out = {"outcome": outcome, "n_obs": len(clean), "n_clusters": g, "controls": ";".join(controls)}
    for group, seed in [("medium", seed_medium), ("high", seed_high)]:
        term = f"post_x_{group}"
        coef, se = float(fit.coef()[term]), float(fit.se()[term])
        try:
            p_wbt = _boot_pvalue(fit.wildboottest(param=term, reps=9999, seed=seed))
        except Exception as exc:
            logger.warning("WCB failed for %s: %s", term, exc)
            p_wbt = np.nan
        out.update({
            f"coef_{group}": coef,
            f"se_{group}": se,
            f"ci_low_{group}": coef - crit * se,
            f"ci_high_{group}": coef + crit * se,
            f"pval_cluster_{group}": float(fit.pvalue()[term]),
            f"pval_wbt_{group}": p_wbt,
        })
    return out


def run_binned_did(panel: pl.DataFrame, post_years: list[int], label: str = "baseline", outcomes: list[str] | None = None, exposure_specs: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = outcomes or ANALYSIS_OUTCOMES
    exposure_specs = exposure_specs or EXPOSURE_SPECS
    rows, assignment_frames = [], []
    for i, spec in enumerate(exposure_specs):
        try:
            did, assignments = build_binned_did_data(panel, post_years, spec)
        except Exception as exc:
            logger.error("Failed to build %s: %s", spec, exc)
            continue
        assignments["label"] = label
        assignment_frames.append(assignments)
        df = did.to_pandas()
        controls = [c for c in BALANCE_CONTROLS if c in df.columns]
        logger.info("Controls [%s]: %s", spec, controls)
        for j, outcome in enumerate(outcomes):
            if outcome not in df.columns:
                continue
            row = run_binned_did_spec(df, outcome, controls, 42 + i * 100 + j * 2, 43 + i * 100 + j * 2)
            row.update({"label": label, "exposure_spec": spec})
            rows.append(row)
    return pd.DataFrame(rows), pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()

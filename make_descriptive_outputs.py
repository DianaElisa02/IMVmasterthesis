"""
make_descriptive_outputs.py
===========================

Creates the key descriptive output for the thesis data section.

Saved outputs:
    output/descriptives/pre_reform_balance_by_exposure_tercile.csv
    output/descriptives/fig_balance_by_exposure_tercile.png
    output/descriptives/fig_exposure_distribution.png

Terminal-only outputs:
    sample overview
    missingness summary
    descriptive statistics
    year-by-year sample counts
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


# =============================================================================
# Paths and settings
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "descriptives"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


EXPOSURE = "exposure_composite_hybrid"

PRE_YEARS = [2017, 2018, 2019]
ESTIMATION_YEARS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

CORE_VARS = [
    "poverty",
    "matdep",
    "income_net_annual",
    "income_equiv_annual",
    "poverty_gap",
    "poverty_gap_sq",
    "hh_size",
]

POTENTIAL_CONTROLS = [
    "head_age",
    "head_female",
    "head_low_edu",
    "head_medium_edu",
    "head_high_edu",
    "head_employed",
    "head_unemployed",
    "head_inactive",
    "n_children",
    "n_adults",
    "single_parent",
    "renting",
    "urban",
    "head_non_eu_born",
]

BALANCE_VARS = [
    "matdep",
    "poverty",
    "income_net_annual",
    "poverty_gap",
    "poverty_gap_sq",
    "hh_size",
]


# =============================================================================
# Helpers
# =============================================================================

def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & w.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def weighted_sd(x: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & w.notna()
    if mask.sum() == 0:
        return np.nan
    x_m = x[mask]
    w_m = w[mask]
    mean = np.average(x_m, weights=w_m)
    var = np.average((x_m - mean) ** 2, weights=w_m)
    return float(np.sqrt(var))


def make_exposure_terciles(df: pd.DataFrame, exposure: str = EXPOSURE) -> pd.DataFrame:
    """
    Assign exposure terciles at the regional level using mean regional exposure.
    Terciles are computed over regions, not households.
    """
    region_exposure = (
        df[["drgn2", exposure]]
        .dropna()
        .groupby("drgn2", as_index=False)[exposure]
        .mean()
        .sort_values(exposure)
    )

    region_exposure["_rank"] = region_exposure[exposure].rank(method="first")
    region_exposure["exposure_tercile"] = pd.qcut(
        region_exposure["_rank"],
        q=3,
        labels=["low", "medium", "high"],
    ).astype(str)

    print("\n=== Regional exposure tercile assignment ===")
    print(region_exposure[["drgn2", exposure, "exposure_tercile"]])

    return df.merge(
        region_exposure[["drgn2", "exposure_tercile"]],
        on="drgn2",
        how="left",
        validate="many_to_one",
    )


# =============================================================================
# Terminal summaries
# =============================================================================

def print_sample_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SAMPLE OVERVIEW")
    print("=" * 80)

    samples = {
        "Full dataset": df,
        "Estimation years excluding 2020": df[df["year"].isin(ESTIMATION_YEARS)],
        "Pre-reform years 2017-2019": df[df["year"].isin(PRE_YEARS)],
        "Post-reform years 2021-2025": df[df["year"].isin([2021, 2022, 2023, 2024, 2025])],
        "COVID-robust post window 2022-2025": df[df["year"].isin([2022, 2023, 2024, 2025])],
    }

    rows = []
    for name, sub in samples.items():
        rows.append({
            "sample": name,
            "n_obs": len(sub),
            "n_unique_households": sub["household_id"].nunique()
            if "household_id" in sub.columns else np.nan,
            "n_regions": sub["drgn2"].nunique(),
            "years": f"{int(sub['year'].min())}-{int(sub['year'].max())}" if len(sub) else "",
        })

    print(pd.DataFrame(rows).to_string(index=False))


def print_year_counts(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("YEAR-BY-YEAR SAMPLE COUNTS")
    print("=" * 80)

    year_counts = (
        df.groupby("year")
        .agg(
            n_obs=("year", "size"),
            n_regions=("drgn2", "nunique"),
        )
        .reset_index()
        .sort_values("year")
    )
    print(year_counts.to_string(index=False))


def print_missingness(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("MISSINGNESS SUMMARY")
    print("=" * 80)

    vars_to_check = [
        "drgn2",
        "year",
        "weight_hh",
        EXPOSURE,
    ] + CORE_VARS + [v for v in POTENTIAL_CONTROLS if v in df.columns]

    vars_to_check = [v for v in vars_to_check if v in df.columns]
    n = len(df)

    rows = []
    for var in vars_to_check:
        n_missing = int(df[var].isna().sum())
        rows.append({
            "variable": var,
            "n_missing": n_missing,
            "share_missing_pct": 100 * n_missing / n,
            "n_nonmissing": int(n - n_missing),
        })

    miss = (
        pd.DataFrame(rows)
        .sort_values(["share_missing_pct", "variable"], ascending=[False, True])
        .reset_index(drop=True)
    )

    print(miss.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


def print_descriptive_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS, ESTIMATION YEARS")
    print("=" * 80)

    est = df[df["year"].isin(ESTIMATION_YEARS)].copy()
    vars_available = [v for v in CORE_VARS + POTENTIAL_CONTROLS if v in est.columns]

    rows = []
    for var in vars_available:
        x = est[var]
        row = {
            "variable": var,
            "n": int(x.notna().sum()),
            "mean_unweighted": x.mean(),
            "sd_unweighted": x.std(),
            "min": x.min(),
            "median": x.median(),
            "max": x.max(),
        }

        if "weight_hh" in est.columns:
            row["mean_weighted"] = weighted_mean(x, est["weight_hh"])
            row["sd_weighted"] = weighted_sd(x, est["weight_hh"])

        rows.append(row)

    desc = pd.DataFrame(rows)
    print(desc.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


# =============================================================================
# Saved table and figures
# =============================================================================

def make_balance_table(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df["year"].isin(PRE_YEARS)].copy()
    pre = make_exposure_terciles(pre, exposure=EXPOSURE)

    balance_vars = [v for v in BALANCE_VARS if v in pre.columns]

    rows = []
    for var in balance_vars:
        row = {"variable": var}

        for group in ["low", "medium", "high"]:
            sub = pre[pre["exposure_tercile"] == group]
            row[f"mean_{group}"] = weighted_mean(sub[var], sub["weight_hh"])
            row[f"n_hh_{group}"] = int(sub[var].notna().sum())

        row["mean_full"] = weighted_mean(pre[var], pre["weight_hh"])
        row["n_hh_full"] = int(pre[var].notna().sum())
        row["diff_high_low"] = row["mean_high"] - row["mean_low"]

        rows.append(row)

    balance = pd.DataFrame(rows)

    path = OUTPUT_DIR / "pre_reform_balance_by_exposure_tercile.csv"
    balance.to_csv(path, index=False)
    logger.info("Saved balance table: %s", path)

    print("\n" + "=" * 80)
    print("PRE-REFORM BALANCE TABLE, WEIGHTED MEANS BY EXPOSURE TERCILE")
    print("=" * 80)
    print(balance.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    return balance


def plot_balance_by_exposure(balance: pd.DataFrame) -> None:
    vars_to_plot = ["poverty", "matdep", "poverty_gap", "poverty_gap_sq"]
    plot_df = balance[balance["variable"].isin(vars_to_plot)].copy()

    labels = {
        "poverty": "At-risk-of-poverty",
        "matdep": "Severe material deprivation",
        "poverty_gap": "Poverty gap",
        "poverty_gap_sq": "Squared poverty gap",
    }

    groups = ["low", "medium", "high"]
    x = np.arange(len(plot_df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for i, group in enumerate(groups):
        ax.bar(
            x + (i - 1) * width,
            plot_df[f"mean_{group}"],
            width=width,
            label=group.capitalize(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(v, v) for v in plot_df["variable"]],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Weighted pre-reform mean")
    ax.set_title("Pre-reform outcomes by exposure tercile")
    ax.legend(title="Exposure tercile")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_balance_by_exposure_tercile.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)


def plot_exposure_distribution(df: pd.DataFrame) -> None:
    region_exp = (
        df[["drgn2", EXPOSURE]]
        .dropna()
        .groupby("drgn2", as_index=False)[EXPOSURE]
        .mean()
        .sort_values(EXPOSURE)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.barh(region_exp["drgn2"].astype(str), region_exp[EXPOSURE])
    ax.set_xlabel("Composite hybrid exposure")
    ax.set_ylabel("Region code")
    ax.set_title("Regional distribution of preferred exposure measure")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_exposure_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logger.info("Reading dataset: %s", INPUT_PATH)

    df = pl.read_parquet(INPUT_PATH).to_pandas()
    logger.info("Loaded dataset: rows=%d, columns=%d", df.shape[0], df.shape[1])

    required = ["year", "drgn2", EXPOSURE]
    missing_required = [v for v in required if v not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required variables: {missing_required}")

    print_sample_overview(df)
    print_year_counts(df)
    print_missingness(df)
    print_descriptive_stats(df)

    balance = make_balance_table(df)
    plot_balance_by_exposure(balance)
    plot_exposure_distribution(df)

    print("\nSaved outputs:")
    print(f"  {OUTPUT_DIR / 'pre_reform_balance_by_exposure_tercile.csv'}")
    print(f"  {OUTPUT_DIR / 'fig_balance_by_exposure_tercile.png'}")
    print(f"  {OUTPUT_DIR / 'fig_exposure_distribution.png'}")


if __name__ == "__main__":
    main()
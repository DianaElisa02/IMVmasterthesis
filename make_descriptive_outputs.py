"""
make_descriptive_outputs.py

Saved outputs:
    output/descriptives/pre_reform_balance_by_exposure_tercile.csv
    output/descriptives/fig_balance_by_exposure_tercile.png
    output/descriptives/fig_exposure_distribution.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

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

TREND_YEARS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

TREND_OUTCOMES = {
    "poverty": "At-risk-of-poverty",
    "matdep": "Severe material deprivation",
    "poverty_gap": "Poverty gap",
    "poverty_gap_sq": "Squared poverty gap",
}

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


REGION_LABELS = {
    11: "Galicia",
    12: "Principado de Asturias",
    13: "Cantabria",
    21: "País Vasco",
    22: "Comunidad Foral de Navarra",
    23: "La Rioja",
    24: "Aragón",
    30: "Comunidad de Madrid",
    41: "Castilla y León",
    42: "Castilla-La Mancha",
    43: "Extremadura",
    51: "Cataluña",
    52: "Comunitat Valenciana",
    53: "Illes Balears",
    61: "Andalucía",
    62: "Región de Murcia",
    63: "Ciudad de Ceuta",
    64: "Ciudad de Melilla",
    70: "Canarias",
}

def plot_exposure_distribution(df: pd.DataFrame) -> None:
    region_exp = (
        df[["drgn2", EXPOSURE]]
        .dropna()
        .groupby("drgn2", as_index=False)[EXPOSURE]
        .mean()
        .sort_values(EXPOSURE)
    )

    region_exp["region_name"] = (
        region_exp["drgn2"]
        .astype(int)
        .map(REGION_LABELS)
    )

    missing = region_exp.loc[region_exp["region_name"].isna(), "drgn2"].tolist()
    if missing:
        raise ValueError(f"Missing region labels for drgn2 codes: {missing}")

    fig, ax = plt.subplots(figsize=(9.5, 6.4))

    bars = ax.barh(
        region_exp["region_name"],
        region_exp[EXPOSURE],
        height=0.68,
        color="#5B8DB8",
        edgecolor="white",
        linewidth=0.8,
    )

    ax.axvline(
        0,
        color="#2F2F2F",
        linewidth=1,
        alpha=0.9,
    )

    ax.set_xlabel("Composite hybrid exposure index", fontsize=10.5)
    ax.set_ylabel("")
    ax.set_title(
        "Regional distribution of the preferred exposure measure",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.grid(axis="x", linestyle="-", alpha=0.18)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0, labelsize=9.5)
    ax.tick_params(axis="x", labelsize=9)

    # Add value labels at the end of each bar
    x_range = region_exp[EXPOSURE].max() - region_exp[EXPOSURE].min()
    offset = 0.025 * x_range

    for bar, value in zip(bars, region_exp[EXPOSURE]):
        if value >= 0:
            x_pos = value + offset
            ha = "left"
        else:
            x_pos = value - offset
            ha = "right"

        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=ha,
            fontsize=8.5,
            color="#333333",
        )

    # Add some horizontal padding so value labels are not cut off
    xmin = region_exp[EXPOSURE].min()
    xmax = region_exp[EXPOSURE].max()
    pad = 0.15 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)

    fig.text(
        0.01,
        0.01,
        "Notes: Higher values indicate larger simulated exposure to the IMV reform relative to the pre-reform RMI system. "
        "The vertical line marks zero exposure.",
        ha="left",
        fontsize=8.3,
        color="#444444",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    path = OUTPUT_DIR / "fig_exposure_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved figure: %s", path)

def make_national_outcome_trend_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate weighted annual national means for the four research outcomes
    over the complete 2017-2025 period, including 2020.
    """
    required = [
        "year",
        "weight_hh",
        *TREND_OUTCOMES.keys(),
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing variables required for national trend figure: {missing}"
        )

    trend_sample = df[
        df["year"].isin(TREND_YEARS)
    ].copy()

    available_years = sorted(
        trend_sample["year"].dropna().astype(int).unique().tolist()
    )

    rows = []

    for year in TREND_YEARS:
        yearly = trend_sample[
            trend_sample["year"].eq(year)
        ]

        row = {"year": year}

        for outcome in TREND_OUTCOMES:
            valid = (
                yearly[outcome].notna()
                & yearly["weight_hh"].notna()
                & np.isfinite(yearly[outcome])
                & np.isfinite(yearly["weight_hh"])
                & yearly["weight_hh"].gt(0)
            )

            if valid.any():
                row[outcome] = weighted_mean(
                    yearly.loc[valid, outcome],
                    yearly.loc[valid, "weight_hh"],
                )
                row[f"n_{outcome}"] = int(valid.sum())
            else:
                row[outcome] = np.nan
                row[f"n_{outcome}"] = 0

        rows.append(row)

    trend_table = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIR
        / "national_outcome_trends_2017_2025.csv"
    )

    trend_table.to_csv(output_path, index=False)

    logger.info(
        "Saved national outcome trend table: %s",
        output_path,
    )

    print("\n" + "=" * 80)
    print("NATIONAL OUTCOME TRENDS, 2017-2025")
    print("=" * 80)

    print(
        trend_table[
            ["year", *TREND_OUTCOMES.keys()]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return trend_table

def plot_national_outcome_trends(
    trend_table: pd.DataFrame,
) -> None:
    """
    Plot continuous national trends for the four outcomes.

    The analytical sample excludes 2020, but the lines connect the 2019 and
    2021 observations. A vertical dotted line identifies the year in which
    the IMV was introduced.
    """

    from matplotlib.ticker import PercentFormatter

    plot_data = (
        trend_table[
            trend_table["year"].isin(
                [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
            )
        ]
        .sort_values("year")
        .copy()
    )

    styles = {
        "poverty": {
            "label": "At-risk-of-poverty",
            "color": "#264653",
            "linewidth": 2.8,
        },
        "matdep": {
            "label": "Severe material deprivation",
            "color": "#457B9D",
            "linewidth": 2.5,
        },
        "poverty_gap": {
            "label": "Poverty gap",
            "color": "#E76F51",
            "linewidth": 2.5,
        },
        "poverty_gap_sq": {
            "label": "Squared poverty gap",
            "color": "#2A9D8F",
            "linewidth": 2.5,
        },
    }

    fig, ax = plt.subplots(figsize=(10.8, 6.3))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw continuous outcome lines.
    for outcome, style in styles.items():
        ax.plot(
            plot_data["year"],
            plot_data[outcome],
            color=style["color"],
            linewidth=style["linewidth"],
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )

        # Show only a small point at the final observation.
        final_value = plot_data.loc[
            plot_data["year"].eq(2025),
            outcome,
        ].iloc[0]

        ax.scatter(
            2025,
            final_value,
            s=24,
            color=style["color"],
            zorder=4,
        )

    # IMV introduction marker.
    ax.axvline(
        x=2020,
        color="#8C8C8C",
        linestyle=(0, (2, 3)),
        linewidth=1.2,
        zorder=1,
    )

    ax.text(
        2020.06,
        0.224,
        "IMV introduced",
        fontsize=9,
        color="#737373",
        ha="left",
        va="top",
    )

    # Direct labels at the right-hand side.
    label_offsets = {
        "poverty": 0.0035,
        "matdep": 0.0025,
        "poverty_gap": -0.0015,
        "poverty_gap_sq": -0.002,
    }

    for outcome, style in styles.items():
        final_value = plot_data.loc[
            plot_data["year"].eq(2025),
            outcome,
        ].iloc[0]

        ax.text(
            2025.15,
            final_value + label_offsets[outcome],
            style["label"],
            color=style["color"],
            fontsize=10,
            fontweight="semibold",
            ha="left",
            va="center",
            clip_on=False,
        )

    # Title and subtitle.
    ax.set_title(
        "National trends in poverty and material deprivation",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=28,
    )

    ax.text(
        0,
        1.025,
        "Weighted annual household means, 2017–2025",
        transform=ax.transAxes,
        fontsize=10,
        color="#666666",
        ha="left",
        va="bottom",
    )

    # Axes.
    ax.set_xlim(2016.8, 2026.6)
    ax.set_ylim(0.025, 0.225)

    ax.set_xticks(
        [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    )

    ax.set_xlabel(
        "Survey year",
        fontsize=10.5,
        labelpad=10,
    )

    ax.set_ylabel(
        "Weighted household mean",
        fontsize=10.5,
        labelpad=10,
    )

    # All outcomes are proportions, so percentages are easier to interpret.
    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1,
            decimals=0,
        )
    )

    # Light horizontal grid only.
    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
        alpha=0.65,
        zorder=0,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    # Minimal borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#B5B5B5")
    ax.spines["bottom"].set_color("#B5B5B5")

    ax.tick_params(
        axis="both",
        colors="#4D4D4D",
        labelsize=9.5,
    )

    # Short methodological note.
    fig.text(
        0.08,
        0.025,
        "Notes: Weighted annual means using ECV household weights. "
        "The analytical sample excludes 2020; the dotted line marks the "
        "introduction of the IMV.",
        fontsize=8.3,
        color="#666666",
        ha="left",
    )

    plt.tight_layout(
        rect=[0.04, 0.07, 0.90, 0.95]
    )

    output_path = (
        OUTPUT_DIR
        / "fig_national_outcome_trends_2017_2025.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    logger.info(
        "Saved national outcome trend figure: %s",
        output_path,
    )

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
    national_trends = make_national_outcome_trend_table(df)
    plot_national_outcome_trends(national_trends)

    print("\nSaved outputs:")
    print(f"  {OUTPUT_DIR / 'pre_reform_balance_by_exposure_tercile.csv'}")
    print(f"  {OUTPUT_DIR / 'fig_balance_by_exposure_tercile.png'}")
    print(f"  {OUTPUT_DIR / 'fig_exposure_distribution.png'}")
    print(f"  {OUTPUT_DIR / 'fig_national_outcome_trends_2017_2025.png'}")


if __name__ == "__main__":
    main()


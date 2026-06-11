"""
plot_outcome_trends_by_tercile.py
=================================
Create one descriptive outcome-trend figure by exposure tercile.

Outputs:
    output/robustness/outcome_trends_by_tercile.csv
    output/robustness/fig_outcome_trends_by_tercile.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from src.constants import ANALYSIS_OUTCOMES
from src.event_study import make_region_terciles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_SPEC = "exposure_composite_hybrid"

OUTCOME_LABELS = {
    "poverty": "At-risk-of-poverty",
    "matdep": "Severe material deprivation",
    "poverty_gap": "Poverty gap",
    "poverty_gap_sq": "Squared poverty gap",
}

TERCILE_ORDER = ["low", "medium", "high"]


def build_trend_data(panel: pl.DataFrame, exposure: str, outcomes: list[str]) -> pd.DataFrame:
    terciles = make_region_terciles(panel, exposure=exposure)

    df = panel.to_pandas()
    df = df.merge(
        terciles[["drgn2", "exposure_tercile"]],
        on="drgn2",
        how="left",
        validate="many_to_one",
    )

    rows = []

    for outcome in outcomes:
        d = df[["year", "drgn2", "exposure_tercile", outcome]].dropna()

        grouped = (
            d.groupby(["year", "exposure_tercile"], observed=True)
            .agg(
                mean_outcome=(outcome, "mean"),
                n_obs=(outcome, "size"),
                n_regions=("drgn2", "nunique"),
            )
            .reset_index()
        )

        grouped["outcome"] = outcome
        grouped["outcome_label"] = OUTCOME_LABELS.get(outcome, outcome)
        grouped["exposure_spec"] = exposure

        rows.append(grouped)

    out = pd.concat(rows, ignore_index=True)

    out["exposure_tercile"] = pd.Categorical(
        out["exposure_tercile"],
        categories=TERCILE_ORDER,
        ordered=True,
    )

    return out.sort_values(["outcome", "exposure_tercile", "year"]).reset_index(drop=True)

def plot_combined_trends(trend_df: pd.DataFrame, output_path: Path) -> None:
    outcomes = [
        o for o in ["poverty", "matdep", "poverty_gap", "poverty_gap_sq"]
        if o in trend_df["outcome"].unique()
    ]

    tercile_labels = {
        "low": "Low exposure",
        "medium": "Medium exposure",
        "high": "High exposure",
    }

    y_axis_labels = {
        "poverty": "Share of households",
        "matdep": "Share of households",
        "poverty_gap": "Mean poverty gap",
        "poverty_gap_sq": "Mean squared poverty gap",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    axes_flat = axes.flatten()

    for idx, outcome in enumerate(outcomes):
        ax = axes_flat[idx]
        d = trend_df[trend_df["outcome"].eq(outcome)].copy()

        for tercile in TERCILE_ORDER:
            sub = d[d["exposure_tercile"].astype(str).eq(tercile)].sort_values("year")
            if sub.empty:
                continue

            ax.plot(
                sub["year"],
                sub["mean_outcome"],
                marker="o",
                markersize=4.5,
                linewidth=1.9,
                label=tercile_labels.get(tercile, tercile.capitalize()),
            )

        ax.axvline(2020, linestyle="--", linewidth=1.1, alpha=0.8)

        ax.set_title(
            OUTCOME_LABELS.get(outcome, outcome),
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel(y_axis_labels.get(outcome, "Mean outcome"))

        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        years = sorted(d["year"].dropna().unique().tolist())
        ax.set_xticks(years)

    for ax in axes_flat[-2:]:
        ax.set_xlabel("Survey year")

    for j in range(len(outcomes), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Exposure group",
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.015),
        )

    fig.suptitle(
        "Descriptive outcome trends by regional exposure group",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.035,
        "Notes: Points report unweighted household-level means by survey year and exposure tercile. "
        "Exposure terciles are based on the preferred hybrid reform-exposure index. "
        "The dashed vertical line marks 2020, the year of IMV introduction.",
        ha="center",
        fontsize=8.5,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved figure: %s", output_path)


def main() -> None:
    logger.info("Reading panel: %s", INPUT_PATH)
    panel = pl.read_parquet(INPUT_PATH)

    panel_cols = set(panel.columns)
    outcomes = [o for o in ANALYSIS_OUTCOMES if o in panel_cols]

    trend_df = build_trend_data(
        panel=panel,
        exposure=PRIMARY_SPEC,
        outcomes=outcomes,
    )

    csv_path = OUTPUT_DIR / "outcome_trends_by_tercile.csv"
    fig_path = OUTPUT_DIR / "fig_outcome_trends_by_tercile.png"

    trend_df.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    plot_combined_trends(trend_df, fig_path)

    logger.info("=== Outcome trend figure complete ===")


if __name__ == "__main__":
    main()
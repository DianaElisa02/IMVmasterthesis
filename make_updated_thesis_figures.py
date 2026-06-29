"""Generate the four updated thesis figures for the preferred tercile design.

Outputs
-------
Descriptive trends:
    output/thesis_figures/outcome_trends_exposure_cov_hybrid.csv
    output/thesis_figures/fig_outcome_trends_exposure_cov_hybrid.png
    output/thesis_figures/outcome_trends_exposure_exp_hybrid.csv
    output/thesis_figures/fig_outcome_trends_exposure_exp_hybrid.png

Tercile event studies:
    output/thesis_figures/event_study_exposure_cov_hybrid.csv
    output/thesis_figures/fig_event_study_exposure_cov_hybrid.png
    output/thesis_figures/event_study_exposure_exp_hybrid.csv
    output/thesis_figures/fig_event_study_exposure_exp_hybrid.png

Run ``python run_event_study.py`` before this script so that the preferred
adjusted event-study coefficient file is current.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from src.binned_did import compute_tercile_assignments
from src.constants import ANALYSIS_OUTCOMES
from src.exposure_specs import EXPOSURE_LABELS, PRIMARY_EXPOSURE_SPECS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PANEL_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
EVENT_STUDY_PATH = (
    BASE_DIR
    / "output"
    / "robustness"
    / "event_study"
    / "event_study_tercile_baseline.csv"
)
OUTPUT_DIR = BASE_DIR / "output" / "thesis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME_LABELS = {
    "poverty": "At-risk-of-poverty",
    "matdep": "Severe material deprivation",
    "poverty_gap": "Poverty gap",
    "poverty_gap_sq": "Squared poverty gap",
}
GROUP_LABELS = {
    "low": "Low exposure",
    "medium": "Medium exposure",
    "high": "High exposure",
}


def _save(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def build_descriptive_trends(panel: pl.DataFrame, exposure_spec: str) -> pd.DataFrame:
    """Create unweighted annual household means using the exact DiD terciles."""
    assignments = compute_tercile_assignments(panel, exposure_spec)
    available_outcomes = [outcome for outcome in ANALYSIS_OUTCOMES if outcome in panel.columns]
    df = panel.select(["year", "drgn2", *available_outcomes]).to_pandas()
    df = df.merge(
        assignments[["drgn2", "exposure_tercile"]],
        on="drgn2",
        how="inner",
        validate="many_to_one",
    )

    frames: list[pd.DataFrame] = []
    for outcome in available_outcomes:
        grouped = (
            df[["year", "drgn2", "exposure_tercile", outcome]]
            .dropna()
            .groupby(["year", "exposure_tercile"], observed=True)
            .agg(
                mean_outcome=(outcome, "mean"),
                n_obs=(outcome, "size"),
                n_regions=("drgn2", "nunique"),
            )
            .reset_index()
        )
        grouped["outcome"] = outcome
        grouped["exposure_spec"] = exposure_spec
        frames.append(grouped)

    return pd.concat(frames, ignore_index=True)


def plot_descriptive_trends(trend_df: pd.DataFrame, exposure_spec: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)

    for ax, outcome in zip(axes.flat, ANALYSIS_OUTCOMES):
        block = trend_df[trend_df["outcome"].eq(outcome)].copy()
        for group in ["low", "medium", "high"]:
            series = block[block["exposure_tercile"].eq(group)].sort_values("year")
            if series.empty:
                continue
            ax.plot(
                series["year"],
                series["mean_outcome"],
                marker="o",
                linewidth=1.8,
                label=GROUP_LABELS[group],
            )

        ax.axvline(2020, linestyle="--", linewidth=1.0)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(sorted(block["year"].dropna().unique()))

    axes[0, 0].set_ylabel("Mean outcome")
    axes[1, 0].set_ylabel("Mean outcome")
    axes[1, 0].set_xlabel("Survey year")
    axes[1, 1].set_xlabel("Survey year")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Descriptive outcome trends by {EXPOSURE_LABELS[exposure_spec].lower()} tercile"
    )
    fig.text(
        0.5,
        0.025,
        "Notes: Points report unweighted household means. Regional terciles match the preferred grouped DiD. "
        "The dashed line marks the 2020 introduction of the IMV; 2020 is excluded from estimation.",
        ha="center",
        fontsize=8.4,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    _save(fig, f"fig_outcome_trends_{exposure_spec}.png")


def load_event_study_results(exposure_spec: str) -> pd.DataFrame:
    event = pd.read_csv(EVENT_STUDY_PATH)
    required = {
        "exposure_spec",
        "outcome",
        "group",
        "year",
        "coef",
        "ci_low",
        "ci_high",
    }
    missing = required.difference(event.columns)
    if missing:
        raise ValueError(f"Event-study output is missing columns: {sorted(missing)}")

    block = event[event["exposure_spec"].eq(exposure_spec)].copy()
    if "control_spec" in block.columns:
        block = block[block["control_spec"].eq("preferred_demographic")]
    if block.empty:
        raise ValueError(f"No preferred tercile event-study results found for {exposure_spec}")
    return block


def plot_event_study(event_df: pd.DataFrame, exposure_spec: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)

    for ax, outcome in zip(axes.flat, ANALYSIS_OUTCOMES):
        block = event_df[event_df["outcome"].eq(outcome)].copy()
        for group in ["medium", "high"]:
            series = block[block["group"].eq(group)].sort_values("year")
            if series.empty:
                continue
            ax.errorbar(
                series["year"],
                series["coef"],
                yerr=[
                    series["coef"] - series["ci_low"],
                    series["ci_high"] - series["coef"],
                ],
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=GROUP_LABELS[group],
            )

        ax.axhline(0, linewidth=1.0)
        ax.axvline(2020, linestyle="--", linewidth=1.0)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(sorted(block["year"].dropna().unique()))

    axes[0, 0].set_ylabel("Coefficient relative to 2019")
    axes[1, 0].set_ylabel("Coefficient relative to 2019")
    axes[1, 0].set_xlabel("Survey year")
    axes[1, 1].set_xlabel("Survey year")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"Preferred adjusted tercile event study: {EXPOSURE_LABELS[exposure_spec]}"
    )
    fig.text(
        0.5,
        0.025,
        "Notes: The omitted reference year is 2019. Error bars are 95% confidence intervals based on "
        "region-clustered standard errors. Models include region and year fixed effects and the preferred demographic controls.",
        ha="center",
        fontsize=8.3,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    _save(fig, f"fig_event_study_{exposure_spec}.png")


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Analysis panel not found: {PANEL_PATH}")
    if not EVENT_STUDY_PATH.exists():
        raise FileNotFoundError(
            f"Event-study results not found: {EVENT_STUDY_PATH}. Run python run_event_study.py first."
        )

    panel = pl.read_parquet(PANEL_PATH)

    for exposure_spec in PRIMARY_EXPOSURE_SPECS:
        trend = build_descriptive_trends(panel, exposure_spec)
        trend.to_csv(
            OUTPUT_DIR / f"outcome_trends_{exposure_spec}.csv",
            index=False,
        )
        plot_descriptive_trends(trend, exposure_spec)

        event = load_event_study_results(exposure_spec)
        event.to_csv(
            OUTPUT_DIR / f"event_study_{exposure_spec}.csv",
            index=False,
        )
        plot_event_study(event, exposure_spec)

    logger.info("Updated thesis figures and source tables saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

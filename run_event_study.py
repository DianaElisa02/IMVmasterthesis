"""
run_event_study.py
==================
Simplified event-study and placebo diagnostics for the IMV DiD design.

This runner is intentionally parsimonious because treatment intensity varies at
the level of a small number of Autonomous Communities.

Main outputs:
    output/robustness/event_study_primary.csv
    output/robustness/event_study_primary_pretrend_summary.csv
    output/robustness/placebo_primary.csv
    output/robustness/event_study_alternative_exposures.csv
    output/robustness/event_study_alternative_pretrend_summary.csv

Figures:
    output/robustness/fig_event_study_primary.png
    output/robustness/fig_placebo_primary.png
    output/robustness/fig_event_study_poverty_by_exposure.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src.constants import (
    ANALYSIS_OUTCOMES,
    BALANCE_CONTROLS,
    EXPOSURE_SPECS,
)
from src.event_study import (
    build_event_study_data,
    run_event_study,
    run_placebo_continuous,
)

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
    "income_net_annual": "Net annual income",
}

EXPOSURE_LABELS = {
    "exposure_composite_hybrid": "Composite hybrid",
    "exposure_exp_hybrid": "Expenditure hybrid",
    "exposure_cov_hybrid": "Coverage hybrid",
    "exposure_composite_sim": "Composite simulated",
    "exposure_admin": "Administrative",
}


# =============================================================================
# Helpers
# =============================================================================

def _write(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        logger.warning("No rows to write: %s", path)
        return
    df.to_csv(path, index=False)
    logger.info("Saved: %s | rows=%d", path, len(df))


def _pretrend_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse event-study rows into one diagnostic-summary row per
    model x exposure x outcome.

    The Wald p-value is a diagnostic, not a definitive test of parallel trends.
    """
    if event_df.empty:
        return pd.DataFrame()

    group_cols = ["model", "exposure_spec", "outcome"]
    summary_cols = [
        "pretrend_wald_stat",
        "pretrend_wald_p",
        "lead_mean",
        "lead_max_abs",
        "post_mean",
        "post_max_abs",
        "lead_to_post_ratio",
        "n_obs",
        "n_clusters",
    ]
    available = [c for c in summary_cols if c in event_df.columns]

    return (
        event_df
        .groupby(group_cols, dropna=False)[available]
        .first()
        .reset_index()
    )


def _plot_one_event_study(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    ylabel: str = "Coefficient",
) -> None:
    d = df.sort_values("year").copy()

    ref = d[d["term"].eq("reference")]
    est = d[~d["term"].eq("reference")].copy()

    pre = est[est["rel_year"] < 0]
    post = est[est["rel_year"] > 0]

    for sub, label in [(pre, "Pre-reform"), (post, "Post-reform")]:
        if sub.empty:
            continue

        yerr = np.vstack([
            sub["coef"] - sub["ci_low"],
            sub["ci_high"] - sub["coef"],
        ])

        ax.errorbar(
            sub["rel_year"],
            sub["coef"],
            yerr=yerr,
            fmt="o",
            capsize=3,
            linewidth=1.2,
            markersize=4,
            label=label,
        )
        ax.plot(sub["rel_year"], sub["coef"], linewidth=1)

    if not ref.empty:
        ax.scatter([0], [0], s=45, marker="s", label="2019 reference")

    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.axvline(1, linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Years relative to 2019")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if "pretrend_wald_p" in d.columns:
        pvals = d["pretrend_wald_p"].dropna()
        if not pvals.empty:
            p = pvals.iloc[0]
            ax.text(
                0.02,
                0.96,
                f"Pretrend diagnostic p={p:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            )


def plot_primary_event_study(event_df: pd.DataFrame, output_path: Path) -> None:
    """
    Option A layout:
        poverty | matdep
        poverty_gap | poverty_gap_sq
    """
    preferred_order = ["poverty", "matdep", "poverty_gap", "poverty_gap_sq"]
    available = set(event_df["outcome"].unique())
    outcomes = [o for o in preferred_order if o in available]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes_flat = axes.flatten()

    for i, outcome in enumerate(outcomes):
        sub = event_df[event_df["outcome"].eq(outcome)]
        _plot_one_event_study(
            axes_flat[i],
            sub,
            OUTCOME_LABELS.get(outcome, outcome),
            ylabel="Effect per exposure unit",
        )

    for j in range(len(outcomes), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3)

    fig.suptitle(
        "Event study — primary exposure: composite hybrid",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", output_path)


def plot_poverty_by_exposure(event_df: pd.DataFrame, output_path: Path) -> None:
    d = event_df[event_df["outcome"].eq("poverty")].copy()

    exposure_order = [
        "exposure_composite_hybrid",
        "exposure_exp_hybrid",
        "exposure_cov_hybrid",
        "exposure_composite_sim",
        "exposure_admin",
    ]
    available = set(d["exposure_spec"].unique())
    exposures = [e for e in exposure_order if e in available]

    if not exposures:
        logger.warning("No poverty event-study rows available for exposure comparison plot.")
        return

    ncols = 2
    nrows = int(np.ceil(len(exposures) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, exposure in enumerate(exposures):
        sub = d[d["exposure_spec"].eq(exposure)]
        _plot_one_event_study(
            axes_flat[i],
            sub,
            EXPOSURE_LABELS.get(exposure, exposure),
            ylabel="Poverty coefficient",
        )

    for j in range(len(exposures), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3)

    fig.suptitle(
        "At-risk-of-poverty event studies by exposure definition",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", output_path)


def plot_placebo_primary(placebo_df: pd.DataFrame, output_path: Path) -> None:
    d = placebo_df.copy()
    d["label"] = d["outcome"].map(OUTCOME_LABELS).fillna(d["outcome"])

    preferred_order = ["poverty", "matdep", "poverty_gap", "poverty_gap_sq"]
    d["order"] = d["outcome"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else 99
    )
    d = d.sort_values("order")

    fig, ax = plt.subplots(figsize=(9, 4.8))

    y = np.arange(len(d))
    xerr = np.vstack([
        d["coef"] - d["ci_low"],
        d["ci_high"] - d["coef"],
    ])

    ax.errorbar(
        d["coef"],
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
        linewidth=1.2,
    )

    ax.axvline(0, linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Placebo coefficient")
    ax.set_title(
        "Placebo DiD — primary exposure, fake treatment in 2019",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)

    x_span = d["ci_high"].max() - d["ci_low"].min()
    offset = 0.03 * x_span if np.isfinite(x_span) and x_span > 0 else 0.001

    for pos, (_, row) in enumerate(d.iterrows()):
        p = row.get("pval_wbt", np.nan)
        txt = f"WCB p={p:.3f}" if not pd.isna(p) else "WCB p=NA"
        ax.text(
            row["ci_high"] + offset,
            pos,
            txt,
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", output_path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logger.info("=== Simplified IMV event-study diagnostics ===")
    logger.info("Input: %s", INPUT_PATH)
    logger.info("Output: %s", OUTPUT_DIR)

    panel = pl.read_parquet(INPUT_PATH)
    panel_cols = set(panel.columns)

    outcomes = [o for o in ANALYSIS_OUTCOMES if o in panel_cols]
    controls = [c for c in BALANCE_CONTROLS if c in panel_cols]

    logger.info("Outcomes used: %s", outcomes)
    logger.info("Controls used: %s", controls)
    logger.info("Primary exposure: %s", PRIMARY_SPEC)

    # -------------------------------------------------------------------------
    # 1. Primary continuous event study
    # -------------------------------------------------------------------------
    primary_rows = []

    df_primary = build_event_study_data(panel, exposure=PRIMARY_SPEC)

    for outcome_idx, outcome in enumerate(outcomes):
        try:
            res = run_event_study(
                df=df_primary,
                outcome=outcome,
                controls=controls,
                exposure=PRIMARY_SPEC,
                region_trends=False,
                model="primary_continuous_event_study",
                seed_base=42 + 10 * outcome_idx,
                run_wcb=True,
            )
            primary_rows.append(res)
        except Exception as exc:
            logger.error("Primary event study failed for %s: %s", outcome, exc)

    primary_event = (
        pd.concat(primary_rows, ignore_index=True)
        if primary_rows
        else pd.DataFrame()
    )

    _write(primary_event, OUTPUT_DIR / "event_study_primary.csv")
    _write(
        _pretrend_summary(primary_event),
        OUTPUT_DIR / "event_study_primary_pretrend_summary.csv",
    )

    if not primary_event.empty:
        plot_primary_event_study(
            primary_event,
            OUTPUT_DIR / "fig_event_study_primary.png",
        )

    # -------------------------------------------------------------------------
    # 2. Primary placebo
    # -------------------------------------------------------------------------
    placebo_rows = []

    for outcome_idx, outcome in enumerate(outcomes):
        try:
            res = run_placebo_continuous(
                panel=panel,
                outcome=outcome,
                exposure=PRIMARY_SPEC,
                controls=controls,
                seed=1000 + outcome_idx,
            )
            placebo_rows.append(res)
        except Exception as exc:
            logger.error("Primary placebo failed for %s: %s", outcome, exc)

    placebo_primary = (
        pd.concat(placebo_rows, ignore_index=True)
        if placebo_rows
        else pd.DataFrame()
    )

    _write(placebo_primary, OUTPUT_DIR / "placebo_primary.csv")

    if not placebo_primary.empty:
        plot_placebo_primary(
            placebo_primary,
            OUTPUT_DIR / "fig_placebo_primary.png",
        )

    # -------------------------------------------------------------------------
    # 3. Alternative exposure event studies
    # -------------------------------------------------------------------------
    # These are sensitivity checks, not separate pass/fail identification tests.
    alt_rows = []

    for exposure_idx, exposure in enumerate(EXPOSURE_SPECS):
        logger.info("Alternative exposure event study: %s", exposure)

        try:
            df_event = build_event_study_data(panel, exposure=exposure)
        except Exception as exc:
            logger.error("Failed to build event data for %s: %s", exposure, exc)
            continue

        for outcome_idx, outcome in enumerate(outcomes):
            try:
                res = run_event_study(
                    df=df_event,
                    outcome=outcome,
                    controls=controls,
                    exposure=exposure,
                    region_trends=False,
                    model="alternative_exposure_event_study",
                    seed_base=2000 + 100 * exposure_idx + 10 * outcome_idx,
                    run_wcb=True,
                )
                alt_rows.append(res)
            except Exception as exc:
                logger.error(
                    "Alternative event study failed for exposure=%s outcome=%s: %s",
                    exposure,
                    outcome,
                    exc,
                )

    alt_event = (
        pd.concat(alt_rows, ignore_index=True)
        if alt_rows
        else pd.DataFrame()
    )

    _write(alt_event, OUTPUT_DIR / "event_study_alternative_exposures.csv")
    _write(
        _pretrend_summary(alt_event),
        OUTPUT_DIR / "event_study_alternative_pretrend_summary.csv",
    )

    if not alt_event.empty:
        plot_poverty_by_exposure(
            alt_event,
            OUTPUT_DIR / "fig_event_study_poverty_by_exposure.png",
        )

    logger.info("=== Simplified diagnostics complete ===")


if __name__ == "__main__":
    main()
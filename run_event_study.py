"""
==================
Orchestrator for the IMV DiD event study.

Reads analysis_dataset.parquet, builds event study data structure,
estimates for each outcome, saves coefficients and plots.

Usage:
  python run_event_study.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import polars as pl
import numpy as np

from src.event_study import build_event_study_data, run_event_study
from src.constants import EVENT_STUDY_REFERENCE_YEAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

BASE_PATH  = Path("/workspaces/IMVmasterthesis")
INPUT_PATH = BASE_PATH / "output" / "analysis_dataset.parquet"
OUTPUT_DIR = BASE_PATH / "output" / "event_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PLOT
# =============================================================================

def plot_event_study(
    coef_table,
    outcome: str,
    output_path: Path,
) -> None:
    """
    Plot event study coefficients with 95% confidence intervals.

    Design choices:
    - Pre-reform coefficients in blue, post-reform in orange
    - Vertical dashed line at 2019 (reference year)
    - Vertical dashed line at 2020.5 (IMV introduction)
    - Horizontal line at zero (null hypothesis)
    - y-axis in percentage points for binary outcomes
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Reference lines
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.axvline(
        EVENT_STUDY_REFERENCE_YEAR, color="grey",
        linewidth=0.8, linestyle=":",
        label=f"Reference year ({EVENT_STUDY_REFERENCE_YEAR})"
    )
    ax.axvline(
        2020.5, color="red", linewidth=0.8, linestyle="--",
        label="IMV introduced (mid-2020)"
    )

    pre  = coef_table[coef_table["year"] <= EVENT_STUDY_REFERENCE_YEAR]
    post = coef_table[coef_table["year"] >  EVENT_STUDY_REFERENCE_YEAR]

    for subset, color, label in [
        (pre,  "steelblue",  "Pre-reform (95% CI, cluster-robust)"),
        (post, "darkorange", "Post-reform (95% CI, cluster-robust)"),
    ]:
        ax.errorbar(
            subset["year"], subset["coef"],
            yerr=1.96 * subset["se"],
            fmt="o", color=color, capsize=4,
            label=label, linewidth=1.5, markersize=5,
            zorder=3,
        )
        ax.plot(
            subset["year"], subset["coef"],
            color=color, linewidth=1.2, zorder=2,
        )

    # Format y-axis as percentage points for binary outcomes
    if outcome in ["matdep", "poverty"]:
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f"{x*100:.1f} pp")
        )

    ax.set_xlabel("Survey year", fontsize=11)
    ax.set_ylabel(
        "Coefficient (pp change in probability)" if outcome in ["matdep", "poverty"]
        else "Coefficient (€ change in income)",
        fontsize=11,
    )
    ax.set_title(
        f"Event study: {outcome}\n"
        f"Year × Exposure interactions (reference: {EVENT_STUDY_REFERENCE_YEAR})",
        fontsize=12,
    )
    ax.set_xticks(sorted(coef_table["year"].tolist()))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax.text(
        0.01, 0.01,
        "CIs are cluster-robust with 15 clusters — indicative only.\n"
        "Use wild cluster bootstrap (Webb, B=9999) for formal inference.",
        transform=ax.transAxes, fontsize=7.5,
        color="grey", va="bottom", style="italic"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Plot saved: %s", output_path)

def main() -> None:
    logger.info("=== IMV DiD — run_event_study.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))

    panel = build_event_study_data(panel)

    for outcome in ["matdep", "poverty", "income_net_annual"]:
        logger.info("--- Outcome: %s ---", outcome)

        coef_table, result, wbt_results = run_event_study(panel, outcome=outcome)

        # Print coefficients
        print(f"\n=== Event study — {outcome} ===")
        print(coef_table.to_string(index=False, float_format="{:.4f}".format))

        # Pre-trend joint F-test (cluster-robust)
        pre_cols = ["yr_2017_x_exposure", "yr_2018_x_exposure"]
        pre_cols_present = [c for c in pre_cols if c in result.params.index]
        if pre_cols_present:
            param_names = result.params.index.tolist()
            R = np.zeros((len(pre_cols_present), len(param_names)))
            for i, col in enumerate(pre_cols_present):
                if col in param_names:
                    R[i, param_names.index(col)] = 1.0
                else:
                    logger.warning("Pre-trend column %s not in params — skipping", col)

            joint_test = result.f_test(R)
            f_stat  = float(np.squeeze(joint_test.fvalue))
            p_value = float(joint_test.pvalue)

            print(f"\nPre-trend joint F-test (2017, 2018 jointly = 0):")
            print(f"  F-stat  = {f_stat:.4f}")
            print(f"  p-value = {p_value:.4f} (cluster-robust, indicative only — 15 clusters)")

            # Wild bootstrap results (computed inside run_event_study)
            if wbt_results:
                print(f"\n  Wild cluster bootstrap (Webb weights, B=9999):")
                for col, p_wbt in wbt_results.items():
                    print(f"    {col}: p = {p_wbt:.4f}")
                print(
                    "  Note: wild bootstrap reports individual p-values.\n"
                    "  Joint pre-trend test uses cluster-robust F-stat above."
                )
            else:
                print("  Wild bootstrap not available — install wildboottest.")

            if p_value > 0.1:
                print("  → Pre-trends not rejected at 10% (cluster-robust)")
            else:
                print("  → WARNING: Pre-trends rejected at 10% (cluster-robust)")

        # Save coefficients
        coef_table.to_csv(
            OUTPUT_DIR / f"event_study_{outcome}.csv", index=False
        )

        # Plot
        plot_event_study(
            coef_table, outcome,
            OUTPUT_DIR / f"event_study_{outcome}.png",
        )

        logger.info("Event study complete. Results saved to %s", OUTPUT_DIR)
if __name__ == "__main__":
    main()
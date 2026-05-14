"""
run_event_study.py
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
import numpy as np
import polars as pl

from src.constants import EVENT_STUDY_REFERENCE_YEAR
from src.event_study import build_event_study_data, run_event_study

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_PATH  = Path("/workspaces/IMVmasterthesis")
INPUT_PATH = BASE_PATH / "output" / "analysis_dataset.parquet"
OUTPUT_DIR = BASE_PATH / "output" / "event_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    - CIs use t_{G-1} critical value (computed in run_event_study) and are
      read directly from ci_low / ci_high columns — not recomputed here
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.axvline(
        EVENT_STUDY_REFERENCE_YEAR, color="grey",
        linewidth=0.8, linestyle=":",
        label=f"Reference year ({EVENT_STUDY_REFERENCE_YEAR})",
    )
    ax.axvline(
        2020.5, color="red", linewidth=0.8, linestyle="--",
        label="IMV introduced (mid-2020)",
    )

    pre  = coef_table[coef_table["year"] <= EVENT_STUDY_REFERENCE_YEAR]
    post = coef_table[coef_table["year"] >  EVENT_STUDY_REFERENCE_YEAR]

    for subset, color, label in [
        (pre,  "steelblue",  "Pre-reform (95% CI)"),
        (post, "darkorange", "Post-reform (95% CI)"),
    ]:
        ci_lower = subset["coef"] - subset["ci_low"]
        ci_upper = subset["ci_high"] - subset["coef"]
        yerr = np.array([ci_lower.values, ci_upper.values])

        ax.errorbar(
            subset["year"], subset["coef"],
            yerr=yerr,
            fmt="o", color=color, capsize=4,
            label=label, linewidth=1.5, markersize=5,
            zorder=3,
        )
        ax.plot(
            subset["year"], subset["coef"],
            color=color, linewidth=1.2, zorder=2,
        )

    if outcome in ["matdep", "poverty"]:
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f"{x*100:.1f} pp")
        )

    ax.set_xlabel("Survey year", fontsize=11)
    ax.set_ylabel(
        "Coefficient (pp change in probability)"
        if outcome in ["matdep", "poverty"]
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
        "95% CIs use t(df=14) critical value (~2.145) on cluster-robust SEs "
        "[15 clusters] — indicative only.\n"
        "Use wild cluster bootstrap p-values (pval_wbt) for formal inference.",
        transform=ax.transAxes, fontsize=7.5,
        color="grey", va="bottom", style="italic",
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

        print(f"\n=== Event study — {outcome} ===")
        print(coef_table.to_string(index=False, float_format="{:.4f}".format))

        pre_cols = ["yr_2017_x_exposure", "yr_2018_x_exposure"]
        pre_cols_present = [c for c in pre_cols if c in result.params.index]

        if pre_cols_present:
            param_names = result.params.index.tolist()

            R = np.zeros((len(pre_cols_present), len(param_names)))
            for i, col in enumerate(pre_cols_present):
                if col in param_names:
                    R[i, param_names.index(col)] = 1.0
                else:
                    logger.warning(
                        "Pre-trend column %s not in params — skipping", col
                    )

            joint_test = result.f_test(R)
            f_stat  = float(np.squeeze(joint_test.fvalue))
            p_value = float(joint_test.pvalue)

            print(f"\nPre-trend joint F-test (2017, 2018 jointly = 0):")
            print(f"  F-stat  = {f_stat:.4f}")
            print(
                f"  p-value = {p_value:.4f} "
                f"(cluster-robust, indicative only — 15 clusters)"
            )

        if wbt_results:
            pre_wbt  = {k: v for k, v in wbt_results.items() if "2017" in k or "2018" in k}
            post_wbt = {k: v for k, v in wbt_results.items() if k not in pre_wbt}

            if pre_wbt:
                print(f"\nWild cluster bootstrap — pre-reform (parallel trends check):")
                for col, p_wbt in pre_wbt.items():
                    flag = "✓" if p_wbt > 0.1 else " WARNING"
                    print(f"    {col}: p = {p_wbt:.4f}  {flag}")

                worst_pre_p = min(pre_wbt.values())
                if worst_pre_p > 0.1:
                    print(
                        "  → Pre-trends not rejected at 10% (WCB individual tests)"
                    )
                else:
                    print(
                        "  → WARNING: At least one pre-trend term rejected at 10% "
                        "(WCB). Parallel trends assumption may be violated."
                    )

            if post_wbt:
                print(f"\nWild cluster bootstrap — post-reform (primary inference):")
                for col, p_wbt in post_wbt.items():
                    flag = "**" if p_wbt < 0.05 else ("*" if p_wbt < 0.1 else "")
                    print(f"    {col}: p = {p_wbt:.4f}  {flag}")

        else:
            print(
                "\n  Wild bootstrap not available — "
                "install wildboottest or check version.\n"
                "  Cluster-robust p-values (pval column) are indicative only."
            )

        coef_table.to_csv(
            OUTPUT_DIR / f"event_study_{outcome}.csv", index=False
        )

        plot_event_study(
            coef_table, outcome,
            OUTPUT_DIR / f"event_study_{outcome}.png",
        )

    logger.info("Event study complete. Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
"""
run_covid_robust.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import polars as pl

from src.constants import (
    ANALYSIS_OUTCOMES,
    COVID_ROBUST_SPECS,
    EVENT_STUDY_REFERENCE_YEAR,
)
from src.covid_robust import build_covid_robust_data, run_covid_robust

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_PATH  = Path("/workspaces/IMVmasterthesis")
INPUT_PATH = BASE_PATH / "output" / "analysis_dataset.parquet"
OUTPUT_DIR = BASE_PATH / "output" / "covid_robust"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_covid_robust(
    coef_table,
    outcome: str,
    spec: dict,
    output_path: Path,
) -> None:
    """
    Plot COVID robustness event study coefficients with 95% CIs.
    Shaded region shows excluded years.
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

    # Shade excluded years
    exclude_years = spec["exclude_years"]
    shade_end = max(exclude_years) + 0.5
    ax.axvspan(
        2020.5, shade_end,
        alpha=0.08, color="red",
        label=f"{spec['label']} (COVID robustness)",
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
        f"COVID robustness — {outcome}\n"
        f"Year × Exposure ({spec['label']}, reference: {EVENT_STUDY_REFERENCE_YEAR})",
        fontsize=12,
    )
    ax.set_xticks(sorted(coef_table["year"].tolist()))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.01, 0.01,
        "CIs are cluster-robust with 15 clusters — indicative only.\n"
        "Use wild cluster bootstrap (Webb, B=9999) for formal inference.",
        transform=ax.transAxes, fontsize=7.5,
        color="grey", va="bottom", style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Plot saved: %s", output_path)

def main() -> None:
    logger.info("=== IMV DiD — run_covid_robust.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))

    for spec_name, spec in COVID_ROBUST_SPECS.items():

        logger.info(
            "--- COVID spec: %s (%s) ---", spec_name, spec["label"]
        )

        # Output subfolder per spec
        spec_dir = OUTPUT_DIR / spec_name
        spec_dir.mkdir(parents=True, exist_ok=True)

        robust, spec_dict = build_covid_robust_data(panel, spec_name)

        for outcome in ANALYSIS_OUTCOMES:
            logger.info("Outcome: %s", outcome)

            coef_table, result, wbt_results = run_covid_robust(
                robust, spec_dict, outcome=outcome
            )

            print(
                f"\n=== COVID robustness [{spec['label']}] — {outcome} ==="
            )
            print(
                coef_table.to_string(index=False, float_format="{:.4f}".format)
            )

            pre_cols = ["yr_2017_x_exposure", "yr_2018_x_exposure"]
            pre_cols_present = [
                c for c in pre_cols if c in result.params.index
            ]
            if pre_cols_present:
                param_names = result.params.index.tolist()
                R = np.zeros((len(pre_cols_present), len(param_names)))
                for i, col in enumerate(pre_cols_present):
                    if col in param_names:
                        R[i, param_names.index(col)] = 1.0
                joint_test = result.f_test(R)
                f_stat  = float(np.squeeze(joint_test.fvalue))
                p_value = float(joint_test.pvalue)
                print(f"\nPre-trend joint F-test (2017, 2018 jointly = 0):")
                print(f"  F-stat  = {f_stat:.4f}")
                print(
                    f"  p-value = {p_value:.4f} "
                    f"(cluster-robust, indicative — 15 clusters)"
                )

            pre_wbt  = {
                k: v for k, v in wbt_results.items()
                if "2017" in k or "2018" in k
            }
            post_wbt = {
                k: v for k, v in wbt_results.items()
                if k not in pre_wbt
            }

            if pre_wbt:
                print(f"\nWild cluster bootstrap — pre-reform:")
                for col, p in pre_wbt.items():
                    mark = "✓" if p > 0.1 else "⚠ WARNING"
                    print(f"    {col}: p = {p:.4f}  {mark}")
                all_pass = all(p > 0.1 for p in pre_wbt.values())
                if all_pass:
                    print(
                        "  → Pre-trends not rejected at 10% (WCB) — "
                        "parallel trends supported"
                    )
                else:
                    print(
                        "  → WARNING: At least one pre-trend rejected "
                        "at 10% (WCB)"
                    )

            if post_wbt:
                print(
                    f"\nWild cluster bootstrap — post-reform "
                    f"({spec['label']}):"
                )
                for col, p in post_wbt.items():
                    stars = (
                        "***" if p < 0.01
                        else "**" if p < 0.05
                        else "*"  if p < 0.10
                        else ""
                    )
                    print(f"    {col}: p = {p:.4f}  {stars}")

            coef_table.to_csv(
                spec_dir / f"covid_robust_{outcome}.csv", index=False
            )

            plot_covid_robust(
                coef_table, outcome, spec_dict,
                spec_dir / f"covid_robust_{outcome}.png",
            )

    logger.info(
        "COVID robustness complete. Results saved to %s", OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
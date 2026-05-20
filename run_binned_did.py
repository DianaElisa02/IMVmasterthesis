"""
run_binned_did.py
=================
Runner for the binned DiD specification.

Runs two post-period windows:
  1. Baseline     : post = 2021-2025
  2. COVID robust : post = 2022-2025

For each window, estimates ANALYSIS_OUTCOMES (matdep, poverty) plus
POVERTY_GAP_OUTCOMES (poverty_gap, poverty_gap_sq) if available.

Reads from analysis_dataset_with_gap.parquet which contains all outcomes.
Run run_poverty_gap.py first to generate the gap columns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.binned_did import build_binned_did_data, run_binned_did
from src.constants import (
    ANALYSIS_OUTCOMES,
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
    EXPOSURE_TERCILES,
    POVERTY_GAP_OUTCOMES,
    REGION_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_PATH  = Path("/workspaces/IMVmasterthesis")
INPUT_PATH = BASE_PATH / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_PATH / "output" / "binned_did"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _stars(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def print_results(df: pd.DataFrame, label: str) -> None:
    if df.empty or "outcome" not in df.columns:
        logger.warning("No results to print for %s", label)
        return

    ref_names = [REGION_NAMES.get(r, str(r)) for r in EXPOSURE_TERCILES["low"]]
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"  Binned DiD — {label}")
    print(f"  Reference: low-exposure tercile ({', '.join(ref_names)})")
    print(f"  Inference: WCB p-value (primary) | CRV1 p-value (auxiliary)")
    print(sep)

    for _, row in df.iterrows():
        stars_M = _stars(row["pval_wbt_medium"])
        stars_H = _stars(row["pval_wbt_high"])

        print(f"\n  Outcome: {row['outcome']}")
        print(f"  {'-'*65}")

        # Medium tercile
        if not pd.isna(row["pval_wbt_medium"]):
            print(
                f"  b_M (medium vs low): {row['coef_medium']:+.4f}  "
                f"SE={row['se_medium']:.4f}  "
                f"CI=[{row['ci_low_medium']:+.4f}, {row['ci_high_medium']:+.4f}]  "
                f"p_CRV1={row['pval_cluster_medium']:.4f}  "
                f"p_WCB={row['pval_wbt_medium']:.4f} {stars_M}"
            )
        else:
            print(
                f"  b_M (medium vs low): {row['coef_medium']:+.4f}  "
                f"SE={row['se_medium']:.4f}  WCB unavailable"
            )

        # High tercile
        if not pd.isna(row["pval_wbt_high"]):
            print(
                f"  b_H (high vs low)  : {row['coef_high']:+.4f}  "
                f"SE={row['se_high']:.4f}  "
                f"CI=[{row['ci_low_high']:+.4f}, {row['ci_high_high']:+.4f}]  "
                f"p_CRV1={row['pval_cluster_high']:.4f}  "
                f"p_WCB={row['pval_wbt_high']:.4f} {stars_H}"
            )
        else:
            print(
                f"  b_H (high vs low)  : {row['coef_high']:+.4f}  "
                f"SE={row['se_high']:.4f}  WCB unavailable"
            )

        # Linearity diagnostic
        ratio_str = (
            f"{row['linearity_ratio']:+.2f}"
            if not pd.isna(row["linearity_ratio"]) else "n/a"
        )
        if not pd.isna(row["linearity_f"]):
            lin_verdict = (
                "linear" if row["linearity_p"] > 0.10 else "NONLINEAR"
            )
            print(
                f"  Linearity b_H/b_M = {ratio_str}  |  "
                f"Wald F={row['linearity_f']:.2f}  "
                f"p={row['linearity_p']:.4f}  [{lin_verdict}]"
            )
        else:
            print(f"  Linearity b_H/b_M = {ratio_str}  |  Wald test unavailable")


def main() -> None:
    logger.info("=== IMV DiD — run_binned_did.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))

    # Check which outcomes are available — poverty gap requires run_poverty_gap.py
    gap_available = [o for o in POVERTY_GAP_OUTCOMES if o in panel.columns]
    if len(gap_available) < len(POVERTY_GAP_OUTCOMES):
        missing_gap = set(POVERTY_GAP_OUTCOMES) - set(gap_available)
        logger.warning(
            "Poverty gap columns not found: %s — run run_poverty_gap.py first. "
            "Proceeding with ANALYSIS_OUTCOMES only.",
            missing_gap,
        )
    outcomes_to_run = ANALYSIS_OUTCOMES + gap_available

    # Log tercile composition
    logger.info("Tercile composition (fixed in constants.py):")
    for grp, regions in EXPOSURE_TERCILES.items():
        names = [REGION_NAMES.get(r, str(r)) for r in regions]
        logger.info("  %s (%d regions): %s", grp, len(regions), names)

    all_results = []

    # ── Baseline (2021-2025) ──────────────────────────────────────────────────
    logger.info("--- Binned DiD: post = 2021-2025 ---")
    did_baseline = build_binned_did_data(panel, post_years=DID_POST_YEARS_BASELINE)
    results_baseline = run_binned_did(
        did_baseline,
        label="baseline_2021_2025",
        outcomes=outcomes_to_run,
    )
    print_results(results_baseline, "Full post-reform period (2021-2025)")
    all_results.append(results_baseline)

    # ── COVID robust (2022-2025) ──────────────────────────────────────────────
    logger.info("--- Binned DiD: post = 2022-2025 ---")
    did_covid = build_binned_did_data(panel, post_years=DID_POST_YEARS_COVID)
    results_covid = run_binned_did(
        did_covid,
        label="covid_robust_2022_2025",
        outcomes=outcomes_to_run,
    )
    print_results(results_covid, "COVID robust — post = 2022-2025")
    all_results.append(results_covid)

    # ── Save ──────────────────────────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "binned_did_results.csv", index=False)
    logger.info("Saved: %s", OUTPUT_DIR / "binned_did_results.csv")

    logger.info("Binned DiD complete. Results: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
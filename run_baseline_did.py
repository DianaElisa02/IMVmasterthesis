"""
run_baseline_did.py
===================
Baseline DiD estimation for the IMV analysis.

Estimates:
  1. Baseline: full post-reform period (2021-2025)
  2. COVID robust: restricted post-reform period (2022-2025)

For each: all ANALYSIS_OUTCOMES × all EXPOSURE_SPECS.

Usage:
  python run_baseline_did.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.baseline_did import build_did_data, run_baseline_did
from src.constants import (
    ANALYSIS_OUTCOMES,
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
    EXPOSURE_SPECS,
)

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
OUTPUT_DIR = BASE_PATH / "output" / "baseline_did"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PRINT RESULTS TABLE
# =============================================================================

def print_results(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*70}")
    print(f"  Baseline DiD — {label}")
    print(f"{'='*70}")
    print(f"  Outcome | Exposure spec | β | SE | CI | p_cluster | p_WCB")
    print(f"{'-'*70}")

    for _, row in df.iterrows():
        stars_wbt = (
            "***" if row["pval_wbt"] < 0.01
            else "**" if row["pval_wbt"] < 0.05
            else "*"  if row["pval_wbt"] < 0.10
            else ""
        ) if not pd.isna(row["pval_wbt"]) else "n/a"

        print(
            f"  {row['outcome']:20s} | "
            f"{row['exposure_spec']:30s} | "
            f"{row['coef']:+.4f} | "
            f"{row['se']:.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"p={row['pval_cluster']:.4f} | "
            f"p={row['pval_wbt']:.4f} {stars_wbt}"
            if not pd.isna(row["pval_wbt"])
            else
            f"  {row['outcome']:20s} | "
            f"{row['exposure_spec']:30s} | "
            f"{row['coef']:+.4f} | "
            f"{row['se']:.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"p={row['pval_cluster']:.4f} | "
            f"WCB unavailable"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("=== IMV DiD — run_baseline_did.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))

    # ── Spec 1: Baseline (2021-2025) ──────────────────────────────────────────
    logger.info("--- Baseline DiD: full post-reform period (2021-2025) ---")
    did_baseline = build_did_data(panel, post_years=DID_POST_YEARS_BASELINE)
    results_baseline = run_baseline_did(did_baseline, label="baseline_2021_2025")
    print_results(results_baseline, "Full post-reform period (2021-2025)")
    results_baseline.to_csv(
        OUTPUT_DIR / "did_baseline_2021_2025.csv", index=False
    )
    logger.info(
        "Saved: %s", OUTPUT_DIR / "did_baseline_2021_2025.csv"
    )

    # ── Spec 2: COVID robust (2022-2025) ──────────────────────────────────────
    logger.info("--- COVID robust DiD: restricted post-reform (2022-2025) ---")
    did_covid = build_did_data(panel, post_years=DID_POST_YEARS_COVID)
    results_covid = run_baseline_did(did_covid, label="covid_robust_2022_2025")
    print_results(results_covid, "COVID robust — post-reform 2022-2025")
    results_covid.to_csv(
        OUTPUT_DIR / "did_covid_robust_2022_2025.csv", index=False
    )
    logger.info(
        "Saved: %s", OUTPUT_DIR / "did_covid_robust_2022_2025.csv"
    )

    # ── Combined table ────────────────────────────────────────────────────────
    combined = pd.concat(
        [results_baseline, results_covid], ignore_index=True
    )
    combined.to_csv(OUTPUT_DIR / "did_all_specs.csv", index=False)
    logger.info("Saved combined: %s", OUTPUT_DIR / "did_all_specs.csv")

    # ── Summary: primary spec only ────────────────────────────────────────────
    primary = EXPOSURE_SPECS[0]
    print(f"\n{'='*70}")
    print(f"  PRIMARY SPEC SUMMARY ({primary})")
    print(f"{'='*70}")
    primary_results = combined[combined["exposure_spec"] == primary]
    for _, row in primary_results.iterrows():
        print(
            f"  {row['label']:30s} | {row['outcome']:20s} | "
            f"β={row['coef']:+.4f} | "
            f"p_WCB={row['pval_wbt']:.4f}"
            if not pd.isna(row["pval_wbt"])
            else
            f"  {row['label']:30s} | {row['outcome']:20s} | "
            f"β={row['coef']:+.4f} | p_WCB=n/a"
        )

    logger.info("Baseline DiD complete. Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
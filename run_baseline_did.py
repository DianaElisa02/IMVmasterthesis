"""
run_baseline_did.py
===================
Runner for the baseline DiD estimation.

Runs two post-period windows separately:
  1. Baseline    : post = 2021–2025 (full post-reform period)
  2. COVID robust: post = 2022–2025 (excludes 2021, peak COVID + low take-up)

For each window, estimates all ANALYSIS_OUTCOMES × EXPOSURE_SPECS.
Results saved as CSV. Primary spec summary printed to console.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.baseline_did import build_did_data, run_baseline_did
from src.constants import (
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
    EXPOSURE_SPECS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_PATH  = Path("/workspaces/IMVmasterthesis")
INPUT_PATH = BASE_PATH / "output" / "analysis_dataset.parquet"
OUTPUT_DIR = BASE_PATH / "output" / "baseline_did"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_SPEC = EXPOSURE_SPECS[0]   # exposure_composite_hybrid


# =============================================================================
# PRINT HELPERS
# =============================================================================

def _stars(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def print_results(df: pd.DataFrame, label: str) -> None:
    """Print a compact results table to stdout."""
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  Baseline DiD — {label}")
    print(f"  Inference: WCB p-value (primary) | CRV1 p-value (auxiliary)")
    print(f"  CI: t(G−1) = t(14) critical value")
    print(sep)
    print(
        f"  {'Outcome':<12} {'Exposure spec':<32} "
        f"{'β':>8} {'SE':>7} {'CI':>22} "
        f"{'p_CRV1':>8} {'p_WCB':>8}"
    )
    print("-" * 78)

    for _, row in df.iterrows():
        stars = _stars(row["pval_wbt"])
        ci_str = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        p_wbt_str = (
            f"{row['pval_wbt']:.4f} {stars}"
            if not pd.isna(row["pval_wbt"])
            else "unavailable"
        )
        print(
            f"  {row['outcome']:<12} {row['exposure_spec']:<32} "
            f"{row['coef']:+8.4f} {row['se']:7.4f} {ci_str:>22} "
            f"{row['pval_cluster']:8.4f} {p_wbt_str:>14}"
        )


def print_primary_summary(df: pd.DataFrame) -> None:
    """Print a one-line summary for the primary exposure spec."""
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  PRIMARY SPEC SUMMARY — {PRIMARY_SPEC}")
    print(sep)

    primary = df[df["exposure_spec"] == PRIMARY_SPEC]
    for _, row in primary.iterrows():
        stars = _stars(row["pval_wbt"])
        p_wbt_str = (
            f"{row['pval_wbt']:.4f} {stars}"
            if not pd.isna(row["pval_wbt"])
            else "n/a"
        )
        print(
            f"  {row['label']:<30} | {row['outcome']:<12} | "
            f"β={row['coef']:+.4f}  SE={row['se']:.4f}  "
            f"p_WCB={p_wbt_str}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("=== IMV DiD — run_baseline_did.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d observations", len(panel))

    # ── 1. Baseline DiD: full post-reform period (2021–2025) ─────────────────
    logger.info("--- Baseline DiD: post = 2021–2025 ---")
    did_baseline = build_did_data(panel, post_years=DID_POST_YEARS_BASELINE)
    results_baseline = run_baseline_did(did_baseline, label="baseline_2021_2025")

    print_results(results_baseline, "Full post-reform period (2021–2025)")
    results_baseline.to_csv(OUTPUT_DIR / "did_baseline_2021_2025.csv", index=False)
    logger.info("Saved: %s", OUTPUT_DIR / "did_baseline_2021_2025.csv")

    # ── 2. COVID robust DiD: restricted post-reform period (2022–2025) ───────
    logger.info("--- COVID robust DiD: post = 2022–2025 ---")
    did_covid = build_did_data(panel, post_years=DID_POST_YEARS_COVID)
    results_covid = run_baseline_did(did_covid, label="covid_robust_2022_2025")

    print_results(results_covid, "COVID robust — post = 2022–2025")
    results_covid.to_csv(OUTPUT_DIR / "did_covid_robust_2022_2025.csv", index=False)
    logger.info("Saved: %s", OUTPUT_DIR / "did_covid_robust_2022_2025.csv")

    # ── 3. Combined output ────────────────────────────────────────────────────
    combined = pd.concat([results_baseline, results_covid], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "did_all_specs.csv", index=False)
    logger.info("Saved combined: %s", OUTPUT_DIR / "did_all_specs.csv")

    # ── 4. Primary spec summary ───────────────────────────────────────────────
    print_primary_summary(combined)

    logger.info("Baseline DiD complete. All results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
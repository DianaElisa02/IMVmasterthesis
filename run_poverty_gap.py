"""
run_poverty_gap.py
==================
Runner for the poverty gap DiD analysis.

Structure
---------
1. Construct poverty_gap, poverty_gap_sq, and weight_person outcomes
   (src/poverty_gap.py)
2. Placebo test — validates parallel trends before causal estimation
3. Baseline DiD — reuses build_did_data() and run_baseline_did() from
   baseline_did.py with outcomes=POVERTY_GAP_OUTCOMES

Changes from original
---------------------
FIX 2 — Weighting consistency
    run_baseline_did() is called with weight_col="weight_person" for poverty
    gap outcomes. This ensures the DiD regressions use person-level weights
    (weight_hh × hh_size), consistent with the person-weighted poverty line
    used to construct the gap. The placebo test is also passed weight_person.

No other estimation logic lives here. All estimation is shared with
run_baseline_did.py through baseline_did.py — no code duplication.
The event study for poverty gap outcomes is handled in run_event_study.py.

Usage
-----
  python run_poverty_gap.py

Note: run this after run_baseline_did.py so the base parquet exists.
The enriched parquet (with gap and weight_person columns) is saved to
output/analysis_dataset_with_gap.parquet for use by run_binned_did.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.baseline_did import build_did_data, run_baseline_did, run_placebo_test
from src.constants import (
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
    EXPOSURE_SPECS,
    POVERTY_GAP_OUTCOMES,
)
from src.poverty_gap import construct_poverty_gap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_THIS_FILE    = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent
INPUT_PATH    = _PROJECT_ROOT / "output" / "analysis_dataset.parquet"
OUTPUT_DIR    = _PROJECT_ROOT / "output" / "poverty_gap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[run_poverty_gap] PROJECT_ROOT = {_PROJECT_ROOT}")
print(f"[run_poverty_gap] INPUT_PATH   = {INPUT_PATH}")

PRIMARY_SPEC = EXPOSURE_SPECS[0]


def _stars(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def print_results(df: pd.DataFrame, label: str) -> None:
    if df.empty or "outcome" not in df.columns:
        logger.warning("No results to print for %s", label)
        return

    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  Poverty Gap DiD — {label}")
    print(f"  Inference: WCB p-value (primary) | CRV1 p-value (auxiliary)")
    print(sep)
    print(
        f"  {'Outcome':<18} {'Exposure spec':<32} "
        f"{'b':>8} {'SE':>7} {'CI':>22} "
        f"{'p_CRV1':>8} {'p_WCB':>8}"
    )
    print("-" * 78)

    for _, row in df.iterrows():
        stars = _stars(row["pval_wbt"])
        ci_str  = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        p_wbt_str = (
            f"{row['pval_wbt']:.4f} {stars}"
            if not pd.isna(row["pval_wbt"])
            else "unavailable"
        )
        print(
            f"  {row['outcome']:<18} {row['exposure_spec']:<32} "
            f"{row['coef']:+8.5f} {row['se']:7.5f} {ci_str:>22} "
            f"{row['pval_cluster']:8.4f} {p_wbt_str:>14}"
        )


def print_placebo(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print(f"\n{'='*60}")
    print("  Placebo test — poverty gap outcomes")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        verdict_icon = "OK" if row["verdict"] == "PASS" else "WARNING"
        print(
            f"  {row['outcome']:<20} b={row['coef']:+.4f}  "
            f"p_WCB={row['pval_wbt']:.4f}  [{verdict_icon}]"
            if not pd.isna(row["pval_wbt"])
            else
            f"  {row['outcome']:<20} b={row['coef']:+.4f}  "
            f"p_WCB=n/a  [{verdict_icon}]"
        )


def main() -> None:
    logger.info("=== IMV — run_poverty_gap.py ===")

    # ── Step 1: Construct poverty gap outcomes ────────────────────────────────
    # construct_poverty_gap also adds weight_person = weight_hh * hh_size
    # and warns if year 2020 is present in the panel (should be excluded
    # upstream per the identification strategy).
    logger.info("Step 1: Constructing poverty gap outcomes")
    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Base panel loaded: %d obs", len(panel))

    panel = construct_poverty_gap(panel)

    logger.info("Step 2: Placebo test (pre-reform falsification)")
    placebo_results = run_placebo_test(
        panel=panel,
        outcomes=POVERTY_GAP_OUTCOMES,
        exposure_spec=PRIMARY_SPEC,
    )
    print_placebo(placebo_results)
    placebo_results.to_csv(OUTPUT_DIR / "placebo_poverty_gap.csv", index=False)
    logger.info("Placebo results saved")

    # Warn if any outcome failed placebo — results reported as appendix only
    failed = placebo_results[placebo_results["verdict"] == "WARNING"]["outcome"].tolist()
    if failed:
        logger.warning(
            "Placebo WARNING for: %s — interpret DiD results with caution", failed
        )


    logger.info("Step 3: DiD — post = 2021-2025")
    did_baseline = build_did_data(panel, post_years=DID_POST_YEARS_BASELINE)
    results_baseline = run_baseline_did(
        did_baseline,
        label="baseline_2021_2025",
        outcomes=POVERTY_GAP_OUTCOMES,
    )
    print_results(results_baseline, "Full post-reform period (2021-2025)")
    results_baseline.to_csv(
        OUTPUT_DIR / "did_poverty_gap_baseline.csv", index=False
    )
    logger.info("Saved: %s", OUTPUT_DIR / "did_poverty_gap_baseline.csv")

    logger.info("Step 4: DiD — post = 2022-2025 (COVID robust)")
    did_covid = build_did_data(panel, post_years=DID_POST_YEARS_COVID)
    results_covid = run_baseline_did(
        did_covid,
        label="covid_robust_2022_2025",
        outcomes=POVERTY_GAP_OUTCOMES,
    )
    print_results(results_covid, "COVID robust — post = 2022-2025")
    results_covid.to_csv(
        OUTPUT_DIR / "did_poverty_gap_covid_robust.csv", index=False
    )
    logger.info("Saved: %s", OUTPUT_DIR / "did_poverty_gap_covid_robust.csv")

    # ── Step 5: Combined output ───────────────────────────────────────────────
    combined = pd.concat([results_baseline, results_covid], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "did_poverty_gap_all_specs.csv", index=False)
    logger.info("Saved combined: %s", OUTPUT_DIR / "did_poverty_gap_all_specs.csv")

    # ── Primary spec summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PRIMARY SPEC SUMMARY ({PRIMARY_SPEC})")
    print(f"{'='*60}")
    primary = combined[combined["exposure_spec"] == PRIMARY_SPEC]
    if primary.empty:
        logger.warning("No primary spec results to summarise")
    else:
        for _, row in primary.iterrows():
            stars = _stars(row["pval_wbt"])
            print(
                f"  {row['label']:<28} | {row['outcome']:<18} | "
                f"b={row['coef']:+.5f}  SE={row['se']:.5f}  "
                f"p_WCB={row['pval_wbt']:.4f} {stars}"
                if not pd.isna(row["pval_wbt"])
                else
                f"  {row['label']:<28} | {row['outcome']:<18} | "
                f"b={row['coef']:+.5f}  SE={row['se']:.5f}  p_WCB=n/a"
            )

    logger.info("Poverty gap analysis complete. Results: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
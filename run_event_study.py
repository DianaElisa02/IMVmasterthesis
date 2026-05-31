"""
run_event_study.py
Runs three analyses:
  1. Event study         — full sample, all event-study years
  2. Placebo test        — pre-reform years only (2017–2019), fake post=2019
  3. Region trends       — event study + drgn2 x year_centered interactions
                           (robustness check, documented as underpowered)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.event_study import (
    build_event_study_data,
    run_event_study,
    run_placebo,
    plot_event_study,
)
from src.constants import (
    ANALYSIS_OUTCOMES,
    BALANCE_CONTROLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "event_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_event_summary(coef_df: pd.DataFrame, outcome: str) -> None:
    sep = "-" * 65
    print(f"\n  Outcome: {outcome}")
    print(f"  {sep}")
    print(f"  {'Year':>6}  {'Rel.yr':>6}  {'Coef':>8}  {'SE':>7}  "
          f"{'p_CRV1':>8}  {'p_WCB':>8}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}")
    for _, row in coef_df.sort_values("year").iterrows():
        stars = ""
        if not pd.isna(row.get("pval_wbt", float("nan"))):
            p = row["pval_wbt"]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        pval_crv1 = f"{row['pval_crv1']:.4f}" if not pd.isna(row["pval_crv1"]) else "  ref "
        pval_wbt  = f"{row['pval_wbt']:.4f} {stars}" if not pd.isna(row.get("pval_wbt", float("nan"))) else "  ref "
        ref_tag   = " ← ref" if row["rel_year"] == 0 else ""
        print(f"  {int(row['year']):>6}  {int(row['rel_year']):>+6}  "
              f"{row['coef']:>+8.4f}  {row['se']:>7.4f}  "
              f"{pval_crv1:>8}  {pval_wbt:>10}{ref_tag}")

    if "pretrend_wald_p" in coef_df.columns:
        wp = coef_df["pretrend_wald_p"].dropna().iloc[0]
        ws = coef_df["pretrend_wald_stat"].dropna().iloc[0]
        verdict = "✓ pre-trends not rejected (p > 0.10)" if wp > 0.10 else "⚠ pre-trends rejected (p ≤ 0.10)"
        print(f"\n  Pre-trend joint Wald: stat={ws:.3f}  p={wp:.4f}  {verdict}")


def print_placebo_summary(results: list[dict]) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  PLACEBO TEST — fake post = 2019, pre-reform years 2017–2019")
    print(f"  H0: beta_placebo = 0  (WCB primary inference)")
    print(sep)
    print(f"  {'Outcome':<12}  {'Coef':>8}  {'SE':>7}  {'p_CRV1':>8}  {'p_WCB':>8}  {'Verdict'}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*20}")
    for r in results:
        p = r["pval_wbt"]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        verdict = "✓ null not rejected" if p > 0.10 else "⚠ significant"
        print(f"  {r['outcome']:<12}  {r['coef']:>+8.4f}  {r['se']:>7.4f}  "
              f"{r['pval_crv1']:>8.4f}  {r['pval_wbt']:>7.4f} {stars:<3}  {verdict}")


def main() -> None:
    logger.info("=== IMV DiD — run_event_study.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))

    controls = [c for c in BALANCE_CONTROLS if c in panel.columns]
    df_event = build_event_study_data(panel)

    print("\n" + "=" * 65)
    print("  EVENT STUDY — primary specification")
    print("  Interaction: year_t x exposure_composite_hybrid")
    print("  Inference: WCB (primary) | CRV1 (auxiliary)")
    print("=" * 65)

    for outcome in ANALYSIS_OUTCOMES:
        logger.info("--- Event study: %s ---", outcome)
        coef_df = run_event_study(
            df_event, outcome=outcome,
            controls=controls,
            output_dir=OUTPUT_DIR,
            region_trends=False,
        )
        print_event_summary(coef_df, outcome)
        plot_event_study(coef_df, outcome, OUTPUT_DIR)

    logger.info("=== Placebo test ===")
    placebo_results = []
    for outcome in ANALYSIS_OUTCOMES:
        logger.info("--- Placebo: %s ---", outcome)
        result = run_placebo(
            panel, outcome=outcome,
            controls=controls,
            output_dir=OUTPUT_DIR,
        )
        placebo_results.append(result)

    print_placebo_summary(placebo_results)

    print("\n" + "=" * 65)
    print("  REGION-SPECIFIC TIME TRENDS (robustness)")
    print("  Note: underpowered with 17 clusters — for documentation only")
    print("=" * 65)

    for outcome in ANALYSIS_OUTCOMES:
        logger.info("--- Region trends: %s ---", outcome)
        try:
            coef_df_rt = run_event_study(
                df_event, outcome=outcome,
                controls=controls,
                output_dir=OUTPUT_DIR,
                region_trends=True,
            )
            print_event_summary(coef_df_rt, outcome)
            plot_event_study(
                coef_df_rt, outcome, OUTPUT_DIR,
                title_suffix=" (region-specific trends)"
            )
        except Exception as e:
            logger.error("Region trends failed for %s: %s", outcome, e)
            print(f"  Region trends failed for {outcome}: {e}")

    logger.info("Event study complete. Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
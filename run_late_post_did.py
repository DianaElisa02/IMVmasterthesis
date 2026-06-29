"""Estimate the IMV DiD using 2023-2025 as the post-treatment period.

The pre-period remains 2017-2019. The script reports the preferred adjusted
tercile DiD for all exposure specifications and an unadjusted comparison for
the two primary hybrid exposures.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.binned_did import build_binned_did_data, run_binned_did, run_binned_did_spec
from src.constants import ANALYSIS_OUTCOMES, EXPOSURE_SPECS
from src.control_specs import PREFERRED_CONTROLS
from src.exposure_specs import PRIMARY_EXPOSURE_SPECS, EXPOSURE_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "binned_did" / "late_post_2023_2025"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POST_YEARS = [2023, 2024, 2025]
LABEL = "late_post_2023_2025"


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def _print_results(results: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    for _, row in results.iterrows():
        print(
            f"{EXPOSURE_LABELS.get(row['exposure_spec'], row['exposure_spec'])} | "
            f"{row['outcome']} | N={int(row['n_obs'])} | "
            f"medium={row['coef_medium']:+.4f} "
            f"(p_WCB={row['pval_wbt_medium']:.4f}{_stars(row['pval_wbt_medium'])}) | "
            f"high={row['coef_high']:+.4f} "
            f"(p_WCB={row['pval_wbt_high']:.4f}{_stars(row['pval_wbt_high'])})"
        )


def _run_unadjusted_primary(panel: pl.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for spec_i, exposure_spec in enumerate(PRIMARY_EXPOSURE_SPECS):
        did, _ = build_binned_did_data(panel, POST_YEARS, exposure_spec)
        df = did.to_pandas()
        for outcome_i, outcome in enumerate(ANALYSIS_OUTCOMES):
            seed = 3000 + spec_i * 100 + outcome_i * 2
            row = run_binned_did_spec(
                df,
                outcome,
                controls=[],
                seed_medium=seed,
                seed_high=seed + 1,
            )
            row.update(
                {
                    "label": LABEL,
                    "exposure_spec": exposure_spec,
                    "control_spec": "unadjusted",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    panel = pl.read_parquet(INPUT_PATH)

    adjusted, assignments = run_binned_did(
        panel=panel,
        post_years=POST_YEARS,
        label=LABEL,
        outcomes=ANALYSIS_OUTCOMES,
        exposure_specs=EXPOSURE_SPECS,
    )
    unadjusted = _run_unadjusted_primary(panel)

    adjusted.to_csv(OUTPUT_DIR / "late_post_adjusted_all_exposures.csv", index=False)
    unadjusted.to_csv(OUTPUT_DIR / "late_post_unadjusted_primary_exposures.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / "late_post_tercile_assignments.csv", index=False)

    primary_adjusted = adjusted[adjusted["exposure_spec"].isin(PRIMARY_EXPOSURE_SPECS)].copy()
    comparison = pd.concat([unadjusted, primary_adjusted], ignore_index=True, sort=False)
    comparison.to_csv(OUTPUT_DIR / "late_post_primary_comparison.csv", index=False)

    _print_results(unadjusted, "2023-2025 POST PERIOD — UNADJUSTED PRIMARY EXPOSURES")
    _print_results(primary_adjusted, "2023-2025 POST PERIOD — PREFERRED ADJUSTED PRIMARY EXPOSURES")
    logger.info("Late-post DiD outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

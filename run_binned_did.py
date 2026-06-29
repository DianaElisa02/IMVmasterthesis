"""Run the baseline tercile DiD specifications."""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import polars as pl

from src.binned_did import run_binned_did
from src.constants import ANALYSIS_OUTCOMES, DID_POST_YEARS_BASELINE, DID_POST_YEARS_COVID, EXPOSURE_SPECS
from src.exposure_specs import PRIMARY_EXPOSURE_SPECS, EXPOSURE_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "binned_did"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def _print_primary(df: pd.DataFrame) -> None:
    print("\nBASELINE TERCILE DiD — CO-PRIMARY HYBRID EXPOSURES")
    for _, row in df[df["exposure_spec"].isin(PRIMARY_EXPOSURE_SPECS)].iterrows():
        print(
            f"{row['label']} | {EXPOSURE_LABELS[row['exposure_spec']]} | {row['outcome']} | "
            f"medium={row['coef_medium']:+.4f} (p_WCB={row['pval_wbt_medium']:.4f}{_stars(row['pval_wbt_medium'])}) | "
            f"high={row['coef_high']:+.4f} (p_WCB={row['pval_wbt_high']:.4f}{_stars(row['pval_wbt_high'])})"
        )


def main() -> None:
    panel = pl.read_parquet(INPUT_PATH)
    outputs, assignments = [], []
    for label, years in [
        ("baseline_2021_2025", DID_POST_YEARS_BASELINE),
        ("covid_robust_2022_2025", DID_POST_YEARS_COVID),
    ]:
        result, groups = run_binned_did(panel, years, label, ANALYSIS_OUTCOMES, EXPOSURE_SPECS)
        outputs.append(result)
        assignments.append(groups)
        result.to_csv(OUTPUT_DIR / f"{label}_all_exposures.csv", index=False)
        groups.to_csv(OUTPUT_DIR / f"{label}_tercile_assignments.csv", index=False)

    combined = pd.concat(outputs, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "binned_did_results.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(OUTPUT_DIR / "tercile_assignments_all_specs.csv", index=False)
    for spec in PRIMARY_EXPOSURE_SPECS:
        combined[combined["exposure_spec"].eq(spec)].to_csv(
            OUTPUT_DIR / f"baseline_tercile_{spec.removeprefix('exposure_')}.csv", index=False
        )
    _print_primary(combined)
    logger.info("Baseline tercile DiD complete: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

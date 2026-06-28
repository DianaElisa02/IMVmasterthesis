"""Run complementary continuous-intensity DiD models.

Outputs include separate standardised models, raw-unit models, and a secondary
joint standardised coverage-benefit specification for two post-period windows.
"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import polars as pl

from src.baseline_did import build_did_data, run_baseline_did
from src.constants import DID_POST_YEARS_BASELINE, DID_POST_YEARS_COVID
from src.continuous_extensions import attach_raw_exposures, run_joint_standardised_model, run_raw_continuous_models
from src.exposure_specs import PRIMARY_EXPOSURE_SPECS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
EXPOSURE_PATH = BASE_DIR / "output" / "exposure" / "exposure_index.csv"
OUTPUT_DIR = BASE_DIR / "output" / "continuous_did"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def _print_primary(df: pd.DataFrame, heading: str) -> None:
    print(f"\n{heading}")
    for _, row in df[df["exposure_spec"].isin(PRIMARY_EXPOSURE_SPECS)].iterrows():
        print(
            f"{row['label']} | {row['exposure_spec']} | {row['outcome']} | "
            f"b={row['coef']:+.5f} SE={row['se']:.5f} "
            f"p_WCB={row['pval_wbt']:.4f}{_stars(row['pval_wbt'])}"
        )


def main() -> None:
    panel = pl.read_parquet(INPUT_PATH)
    panel = attach_raw_exposures(panel, EXPOSURE_PATH)
    standardised_frames, raw_frames, joint_frames = [], [], []

    windows = [
        ("continuous_2021_2025", DID_POST_YEARS_BASELINE),
        ("continuous_covid_robust_2022_2025", DID_POST_YEARS_COVID),
    ]

    for label, years in windows:
        did = build_did_data(panel, post_years=years)
        standardised = run_baseline_did(did, label=label)
        standardised["scale"] = "standardised_separate"
        raw = run_raw_continuous_models(panel, years, f"{label}_raw")
        joint = run_joint_standardised_model(panel, years, f"{label}_joint")

        standardised.to_csv(OUTPUT_DIR / f"{label}_standardised_separate.csv", index=False)
        raw.to_csv(OUTPUT_DIR / f"{label}_raw_units.csv", index=False)
        joint.to_csv(OUTPUT_DIR / f"{label}_joint_standardised.csv", index=False)

        standardised_frames.append(standardised)
        raw_frames.append(raw)
        joint_frames.append(joint)

    standardised_all = pd.concat(standardised_frames, ignore_index=True)
    raw_all = pd.concat(raw_frames, ignore_index=True)
    joint_all = pd.concat(joint_frames, ignore_index=True)
    standardised_all.to_csv(OUTPUT_DIR / "continuous_did_standardised_all.csv", index=False)
    raw_all.to_csv(OUTPUT_DIR / "continuous_did_raw_all.csv", index=False)
    joint_all.to_csv(OUTPUT_DIR / "continuous_did_joint_all.csv", index=False)
    pd.concat([standardised_all, raw_all, joint_all], ignore_index=True, sort=False).to_csv(
        OUTPUT_DIR / "continuous_did_all_models.csv", index=False
    )

    _print_primary(standardised_all, "SEPARATE STANDARDISED CONTINUOUS MODELS")
    logger.info("Continuous DiD outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

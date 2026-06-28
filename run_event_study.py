"""Run baseline tercile and complementary continuous event-study diagnostics."""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import polars as pl

from src.constants import ANALYSIS_OUTCOMES, EXPOSURE_SPECS
from src.event_study import run_continuous_event_study, run_tercile_event_study
from src.exposure_specs import PRIMARY_EXPOSURE_SPECS, EXPOSURE_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "robustness" / "event_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _diagnostic_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = [c for c in ["model", "exposure_spec", "outcome", "group"] if c in df.columns]
    cols = [c for c in ["pretrend_wald_stat", "pretrend_wald_p", "pretrend_group_wald_stat", "pretrend_group_wald_p", "n_obs", "n_clusters"] if c in df.columns]
    out = df.groupby(group_cols, dropna=False)[cols].first().reset_index()
    out["exposure_label"] = out["exposure_spec"].map(EXPOSURE_LABELS).fillna(out["exposure_spec"])
    out["interpretation"] = out["pretrend_wald_p"].apply(
        lambda p: "Diagnostic unavailable" if pd.isna(p) else (
            "No evidence of differential pre-trends at the 10% level" if p > 0.10
            else "Evidence inconsistent with parallel pre-trends at the 10% level"
        )
    )
    return out


def main() -> None:
    panel = pl.read_parquet(INPUT_PATH)
    tercile_frames, continuous_frames, assignments = [], [], []

    for spec in PRIMARY_EXPOSURE_SPECS:
        logger.info("Baseline tercile event study: %s", spec)
        event, groups = run_tercile_event_study(panel, spec, ANALYSIS_OUTCOMES)
        tercile_frames.append(event)
        assignments.append(groups)

    for spec in EXPOSURE_SPECS:
        if spec not in panel.columns:
            logger.warning("Skipping absent exposure: %s", spec)
            continue
        logger.info("Complementary continuous event study: %s", spec)
        continuous_frames.append(run_continuous_event_study(panel, spec, ANALYSIS_OUTCOMES))

    tercile = pd.concat(tercile_frames, ignore_index=True) if tercile_frames else pd.DataFrame()
    continuous = pd.concat(continuous_frames, ignore_index=True) if continuous_frames else pd.DataFrame()
    groups = pd.concat(assignments, ignore_index=True) if assignments else pd.DataFrame()

    tercile.to_csv(OUTPUT_DIR / "event_study_tercile_baseline.csv", index=False)
    continuous.to_csv(OUTPUT_DIR / "event_study_continuous_complementary.csv", index=False)
    groups.to_csv(OUTPUT_DIR / "event_study_tercile_assignments.csv", index=False)
    _diagnostic_summary(tercile).to_csv(OUTPUT_DIR / "event_study_tercile_pretrend_summary.csv", index=False)
    _diagnostic_summary(continuous).to_csv(OUTPUT_DIR / "event_study_continuous_pretrend_summary.csv", index=False)
    logger.info("Event-study outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""
run_event_study.py
==================
Runs validity and robustness checks for the IMV DiD design.

Outputs are written to:
    output/robustness/

Generated files:
    1. event_study_continuous.csv
    2. event_study_continuous_pretrend_summary.csv
    3. placebo_continuous.csv
    4. event_study_terciles.csv
    5. event_study_terciles_pretrend_summary.csv
    6. placebo_terciles.csv
    7. event_study_region_trends.csv
    8. event_study_region_trends_pretrend_summary.csv
    9. exposure_tercile_assignments.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.constants import (
    ANALYSIS_OUTCOMES,
    BALANCE_CONTROLS,
    EXPOSURE_SPECS,
)
from src.event_study import (
    build_event_study_data,
    build_tercile_event_study_data,
    make_region_terciles,
    run_event_study,
    run_event_study_terciles,
    run_placebo_continuous,
    run_placebo_terciles,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_SPEC = EXPOSURE_SPECS[0]


def _write(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        logger.warning("No rows to write: %s", path)
        return
    df.to_csv(path, index=False)
    logger.info("Saved: %s | rows=%d", path, len(df))


def _pretrend_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse event-study rows into one pretrend-summary row per
    model x exposure x outcome, and group where relevant.
    """
    if event_df.empty:
        return pd.DataFrame()

    group_cols = ["model", "exposure_spec", "outcome"]
    if "group" in event_df.columns:
        group_cols.append("group")

    summary_cols = [
        "pretrend_wald_stat",
        "pretrend_wald_p",
        "lead_mean",
        "lead_max_abs",
        "n_obs",
        "n_clusters",
    ]
    available = [c for c in summary_cols if c in event_df.columns]

    summary = (
        event_df
        .groupby(group_cols, dropna=False)[available]
        .first()
        .reset_index()
    )
    return summary


def main() -> None:
    logger.info("=== IMV robustness/event-study checks ===")
    logger.info("Input: %s", INPUT_PATH)
    logger.info("Output directory: %s", OUTPUT_DIR)

    panel = pl.read_parquet(INPUT_PATH)

    logger.info("Panel rows: %d", len(panel))
    logger.info("Outcomes: %s", ANALYSIS_OUTCOMES)
    logger.info("Exposure specs: %s", EXPOSURE_SPECS)

    panel_cols = set(panel.columns)
    controls = [c for c in BALANCE_CONTROLS if c in panel_cols]

    logger.info("Controls used: %s", controls)

    # -------------------------------------------------------------------------
    # 1. Continuous event study — all exposure specs x all outcomes
    # -------------------------------------------------------------------------
    continuous_event_rows = []

    for exposure_idx, exposure in enumerate(EXPOSURE_SPECS):
        logger.info("Continuous event study: exposure=%s", exposure)

        try:
            df_event = build_event_study_data(panel, exposure=exposure)
        except Exception as exc:
            logger.error("Failed to build continuous event data for %s: %s", exposure, exc)
            continue

        for outcome_idx, outcome in enumerate(ANALYSIS_OUTCOMES):
            if outcome not in panel_cols:
                logger.warning("Outcome not found, skipping: %s", outcome)
                continue

            try:
                result = run_event_study(
                    df=df_event,
                    outcome=outcome,
                    controls=controls,
                    exposure=exposure,
                    region_trends=False,
                    model="continuous_event_study",
                    seed_base=42 + 100 * exposure_idx + 10 * outcome_idx,
                    run_wcb=True,
                )
                continuous_event_rows.append(result)
            except Exception as exc:
                logger.error(
                    "Continuous event study failed: exposure=%s outcome=%s error=%s",
                    exposure,
                    outcome,
                    exc,
                )

    continuous_event = (
        pd.concat(continuous_event_rows, ignore_index=True)
        if continuous_event_rows else pd.DataFrame()
    )

    _write(continuous_event, OUTPUT_DIR / "event_study_continuous.csv")
    _write(
        _pretrend_summary(continuous_event),
        OUTPUT_DIR / "event_study_continuous_pretrend_summary.csv",
    )

    # -------------------------------------------------------------------------
    # 2. Continuous placebo — all exposure specs x all outcomes
    # -------------------------------------------------------------------------
    continuous_placebo_rows = []

    for exposure_idx, exposure in enumerate(EXPOSURE_SPECS):
        logger.info("Continuous placebo: exposure=%s", exposure)

        for outcome_idx, outcome in enumerate(ANALYSIS_OUTCOMES):
            if outcome not in panel_cols:
                logger.warning("Outcome not found, skipping: %s", outcome)
                continue

            try:
                result = run_placebo_continuous(
                    panel=panel,
                    outcome=outcome,
                    exposure=exposure,
                    controls=controls,
                    seed=1000 + 100 * exposure_idx + outcome_idx,
                )
                continuous_placebo_rows.append(result)
            except Exception as exc:
                logger.error(
                    "Continuous placebo failed: exposure=%s outcome=%s error=%s",
                    exposure,
                    outcome,
                    exc,
                )

    continuous_placebo = (
        pd.concat(continuous_placebo_rows, ignore_index=True)
        if continuous_placebo_rows else pd.DataFrame()
    )

    _write(continuous_placebo, OUTPUT_DIR / "placebo_continuous.csv")

    # -------------------------------------------------------------------------
    # 3. Tercile assignments — all exposure specs
    # -------------------------------------------------------------------------
    tercile_assignment_rows = []

    for exposure in EXPOSURE_SPECS:
        try:
            terciles = make_region_terciles(panel, exposure=exposure)
            terciles["exposure_spec"] = exposure
            tercile_assignment_rows.append(terciles)
        except Exception as exc:
            logger.error("Tercile assignment failed for %s: %s", exposure, exc)

    tercile_assignments = (
        pd.concat(tercile_assignment_rows, ignore_index=True)
        if tercile_assignment_rows else pd.DataFrame()
    )

    _write(tercile_assignments, OUTPUT_DIR / "exposure_tercile_assignments.csv")

    # -------------------------------------------------------------------------
    # 4. Tercile event study — all exposure specs x all outcomes
    # -------------------------------------------------------------------------
    tercile_event_rows = []

    for exposure_idx, exposure in enumerate(EXPOSURE_SPECS):
        logger.info("Tercile event study: exposure=%s", exposure)

        try:
            df_tercile, _ = build_tercile_event_study_data(panel, exposure=exposure)
        except Exception as exc:
            logger.error("Failed to build tercile event data for %s: %s", exposure, exc)
            continue

        for outcome_idx, outcome in enumerate(ANALYSIS_OUTCOMES):
            if outcome not in panel_cols:
                logger.warning("Outcome not found, skipping: %s", outcome)
                continue

            try:
                result = run_event_study_terciles(
                    df=df_tercile,
                    outcome=outcome,
                    exposure=exposure,
                    controls=controls,
                    seed_base=2000 + 100 * exposure_idx + 10 * outcome_idx,
                )
                tercile_event_rows.append(result)
            except Exception as exc:
                logger.error(
                    "Tercile event study failed: exposure=%s outcome=%s error=%s",
                    exposure,
                    outcome,
                    exc,
                )

    tercile_event = (
        pd.concat(tercile_event_rows, ignore_index=True)
        if tercile_event_rows else pd.DataFrame()
    )

    _write(tercile_event, OUTPUT_DIR / "event_study_terciles.csv")
    _write(
        _pretrend_summary(tercile_event),
        OUTPUT_DIR / "event_study_terciles_pretrend_summary.csv",
    )

    # -------------------------------------------------------------------------
    # 5. Tercile placebo — all exposure specs x all outcomes
    # -------------------------------------------------------------------------
    tercile_placebo_rows = []

    for exposure_idx, exposure in enumerate(EXPOSURE_SPECS):
        logger.info("Tercile placebo: exposure=%s", exposure)

        for outcome_idx, outcome in enumerate(ANALYSIS_OUTCOMES):
            if outcome not in panel_cols:
                logger.warning("Outcome not found, skipping: %s", outcome)
                continue

            try:
                result = run_placebo_terciles(
                    panel=panel,
                    outcome=outcome,
                    exposure=exposure,
                    controls=controls,
                    seed_base=3000 + 100 * exposure_idx + 10 * outcome_idx,
                )
                tercile_placebo_rows.append(result)
            except Exception as exc:
                logger.error(
                    "Tercile placebo failed: exposure=%s outcome=%s error=%s",
                    exposure,
                    outcome,
                    exc,
                )

    tercile_placebo = (
        pd.concat(tercile_placebo_rows, ignore_index=True)
        if tercile_placebo_rows else pd.DataFrame()
    )

    _write(tercile_placebo, OUTPUT_DIR / "placebo_terciles.csv")

    # -------------------------------------------------------------------------
    # 6. Region-specific trend event study — primary exposure only
    # -------------------------------------------------------------------------
    # To avoid too many tables, region-trend robustness is run only for the
    # primary exposure across all outcomes.
    region_trend_rows = []

    try:
        df_event_primary = build_event_study_data(panel, exposure=PRIMARY_SPEC)

        for outcome_idx, outcome in enumerate(ANALYSIS_OUTCOMES):
            if outcome not in panel_cols:
                logger.warning("Outcome not found, skipping: %s", outcome)
                continue

            try:
                result = run_event_study(
                    df=df_event_primary,
                    outcome=outcome,
                    controls=controls,
                    exposure=PRIMARY_SPEC,
                    region_trends=True,
                    model="continuous_event_study_region_trends",
                    seed_base=4000 + 10 * outcome_idx,
                    run_wcb=False,
                )
                region_trend_rows.append(result)
            except Exception as exc:
                logger.error(
                    "Region-trend event study failed: outcome=%s error=%s",
                    outcome,
                    exc,
                )

    except Exception as exc:
        logger.error("Failed to build primary event-study data for region trends: %s", exc)

    region_trends = (
        pd.concat(region_trend_rows, ignore_index=True)
        if region_trend_rows else pd.DataFrame()
    )

    _write(region_trends, OUTPUT_DIR / "event_study_region_trends.csv")
    _write(
        _pretrend_summary(region_trends),
        OUTPUT_DIR / "event_study_region_trends_pretrend_summary.csv",
    )

    logger.info("=== Robustness checks complete ===")


if __name__ == "__main__":
    main()
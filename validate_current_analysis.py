"""Validate that current analysis outputs match the preferred specification."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from src.binned_did import compute_tercile_assignments
from src.constants import ANALYSIS_N_CLUSTERS
from src.control_specs import PREFERRED_CONTROLS, add_preferred_control_groups
from src.exposure_specs import PRIMARY_EXPOSURE_SPECS

BASE_DIR = Path(__file__).resolve().parent
PANEL_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
BINNED_PATH = BASE_DIR / "output" / "binned_did" / "binned_did_results.csv"
EVENT_PATH = BASE_DIR / "output" / "robustness" / "event_study" / "event_study_tercile_baseline.csv"
FIGURE_DIR = BASE_DIR / "output" / "thesis_figures"


def main() -> None:
    panel = add_preferred_control_groups(pl.read_parquet(PANEL_PATH))

    missing_controls = [control for control in PREFERRED_CONTROLS if control not in panel.columns]
    if missing_controls:
        raise AssertionError(f"Preferred controls missing: {missing_controls}")

    n_regions = panel.select(pl.col("drgn2").n_unique()).item()
    if n_regions != ANALYSIS_N_CLUSTERS:
        raise AssertionError(f"Expected {ANALYSIS_N_CLUSTERS} regions, found {n_regions}")

    for exposure_spec in PRIMARY_EXPOSURE_SPECS:
        assignments = compute_tercile_assignments(panel, exposure_spec)
        if assignments["drgn2"].nunique() != ANALYSIS_N_CLUSTERS:
            raise AssertionError(f"Unexpected region count for {exposure_spec}")
        if set(assignments["exposure_tercile"]) != {"low", "medium", "high"}:
            raise AssertionError(f"Incomplete tercile assignment for {exposure_spec}")

    binned = pd.read_csv(BINNED_PATH)
    if "control_spec" not in binned.columns:
        raise AssertionError("Binned output lacks control_spec")
    if not binned["control_spec"].eq("preferred_demographic").all():
        raise AssertionError("Binned output contains non-preferred control specifications")

    event = pd.read_csv(EVENT_PATH)
    if "control_spec" not in event.columns:
        raise AssertionError("Event-study output lacks control_spec")
    if not event["control_spec"].eq("preferred_demographic").all():
        raise AssertionError("Event-study output contains non-preferred control specifications")

    expected_figures = [
        FIGURE_DIR / f"fig_outcome_trends_{spec}.png"
        for spec in PRIMARY_EXPOSURE_SPECS
    ] + [
        FIGURE_DIR / f"fig_event_study_{spec}.png"
        for spec in PRIMARY_EXPOSURE_SPECS
    ]
    missing_figures = [str(path) for path in expected_figures if not path.exists()]
    if missing_figures:
        raise AssertionError(f"Expected figures missing: {missing_figures}")

    print("Validation passed:")
    print(f"- {n_regions} regions")
    print(f"- preferred controls present: {PREFERRED_CONTROLS}")
    print("- binned and event-study outputs labelled preferred_demographic")
    print("- all four current thesis figures found")


if __name__ == "__main__":
    main()

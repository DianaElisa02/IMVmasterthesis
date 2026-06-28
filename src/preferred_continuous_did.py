"""Continuous DiD estimation using the preferred demographic controls."""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl

from src.baseline_did import run_did_spec
from src.constants import ANALYSIS_OUTCOMES, EXPOSURE_SPECS
from src.control_specs import PREFERRED_CONTROLS, add_preferred_control_groups

logger = logging.getLogger(__name__)


def run_preferred_continuous_did(
    did: pl.DataFrame,
    label: str = "baseline",
    outcomes: list[str] | None = None,
) -> pd.DataFrame:
    """Estimate separate continuous-exposure models with preferred controls."""
    outcomes = outcomes or ANALYSIS_OUTCOMES
    did = add_preferred_control_groups(did)
    df = did.to_pandas()

    missing_controls = [control for control in PREFERRED_CONTROLS if control not in df.columns]
    if missing_controls:
        raise ValueError(f"Preferred controls missing from analysis data: {missing_controls}")

    rows: list[dict] = []
    for outcome_idx, outcome in enumerate(outcomes):
        if outcome not in df.columns:
            logger.warning("Outcome '%s' not in panel -- skipping", outcome)
            continue
        for spec_idx, exposure_spec in enumerate(EXPOSURE_SPECS):
            interaction = f"post_x_{exposure_spec}"
            if interaction not in df.columns:
                logger.warning("Interaction '%s' not in panel -- skipping", interaction)
                continue
            row = run_did_spec(
                df=df,
                outcome=outcome,
                exposure_spec=exposure_spec,
                controls=PREFERRED_CONTROLS,
                seed=42 + outcome_idx * len(EXPOSURE_SPECS) + spec_idx,
            )
            row.update(
                {
                    "label": label,
                    "controls": ";".join(PREFERRED_CONTROLS),
                    "control_spec": "preferred_demographic",
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)

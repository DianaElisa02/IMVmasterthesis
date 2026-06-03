"""
diagnose_collinearity.py
========================
Diagnostic script to identify which controls create multicollinearity
in the continuous event-study specification.

Run from the repository root:

    python diagnose_collinearity.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl
import pyfixest as pf

from src.constants import (
    ANALYSIS_OUTCOMES,
    BALANCE_CONTROLS,
    EVENT_STUDY_REFERENCE_YEAR,
    EXPOSURE_SPECS,
)
from src.event_study import build_event_study_data

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
REF_YEAR = EVENT_STUDY_REFERENCE_YEAR


def _event_terms_from_fit(fit) -> list[str]:
    """
    Return estimated event-study coefficient names.
    """
    terms = []
    for name in fit.coef().index.tolist():
        if "year" in name and "_exposure" in name:
            terms.append(name)
        elif "_exposure" in name:
            terms.append(name)
    return terms


def _run_model(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    label: str,
) -> dict:
    """
    Estimate one event-study model and record which coefficients survive.
    """
    work = df.copy()

    if "_exposure" not in work.columns:
        raise ValueError("Expected '_exposure' column in event-study data.")

    keep_cols = [outcome, "drgn2", "year", "_exposure"] + controls
    keep_cols = [c for c in keep_cols if c in work.columns]

    work = work[keep_cols].dropna().reset_index(drop=True)

    ctrl_str = (" + " + " + ".join(controls)) if controls else ""

    formula = (
        f"{outcome} ~ i(year, _exposure, ref={REF_YEAR})"
        + ctrl_str
        + " | drgn2 + year"
    )

    try:
        fit = pf.feols(
            formula,
            data=work,
            vcov={"CRV1": "drgn2"},
        )

        coef_names = fit.coef().index.tolist()
        event_terms = _event_terms_from_fit(fit)

        estimated_years = []
        for term in event_terms:
            for year in sorted(work["year"].dropna().unique()):
                year_int = int(year)
                if str(year_int) in term:
                    estimated_years.append(year_int)

        estimated_years = sorted(set(estimated_years))

        return {
            "spec": label,
            "outcome": outcome,
            "controls": ", ".join(controls) if controls else "none",
            "n_controls": len(controls),
            "n_obs": len(work),
            "n_clusters": int(work["drgn2"].nunique()),
            "formula": formula,
            "estimated_event_terms": " | ".join(event_terms),
            "estimated_event_years": ", ".join(map(str, estimated_years)),
            "all_coefficients": " | ".join(coef_names),
            "error": "",
        }

    except Exception as exc:
        return {
            "spec": label,
            "outcome": outcome,
            "controls": ", ".join(controls) if controls else "none",
            "n_controls": len(controls),
            "n_obs": len(work),
            "n_clusters": int(work["drgn2"].nunique()) if "drgn2" in work.columns else None,
            "formula": formula,
            "estimated_event_terms": "",
            "estimated_event_years": "",
            "all_coefficients": "",
            "error": str(exc),
        }


def diagnose_controls(
    df_event: pd.DataFrame,
    outcome: str,
    controls: list[str],
) -> pd.DataFrame:
    """
    Run event-study models with controls added sequentially.

    The first row is no controls. Each next row adds one more control.
    The final row repeats the full control set.
    """
    rows = []

    rows.append(
        _run_model(
            df=df_event,
            outcome=outcome,
            controls=[],
            label="00_no_controls",
        )
    )

    running_controls: list[str] = []

    for idx, control in enumerate(controls, start=1):
        running_controls.append(control)

        rows.append(
            _run_model(
                df=df_event,
                outcome=outcome,
                controls=running_controls.copy(),
                label=f"{idx:02d}_add_{control}",
            )
        )

    rows.append(
        _run_model(
            df=df_event,
            outcome=outcome,
            controls=controls,
            label="full_controls",
        )
    )

    return pd.DataFrame(rows)


def check_mechanical_collinearity(panel: pl.DataFrame) -> pd.DataFrame:
    """
    Check obvious mechanical relationships among household-size variables.
    """
    needed = {"hh_size", "n_children", "n_adults"}
    if not needed.issubset(set(panel.columns)):
        return pd.DataFrame([{
            "check": "hh_size_vs_children_adults",
            "available": False,
            "message": "One or more of hh_size, n_children, n_adults is missing.",
        }])

    df = panel.select(["hh_size", "n_children", "n_adults"]).to_pandas().dropna()
    diff = df["hh_size"] - df["n_children"] - df["n_adults"]

    return pd.DataFrame([{
        "check": "hh_size_minus_n_children_minus_n_adults",
        "available": True,
        "n_obs": len(diff),
        "min": diff.min(),
        "max": diff.max(),
        "mean": diff.mean(),
        "all_zero": bool((diff == 0).all()),
        "message": (
            "If all_zero is True, do not include hh_size, n_children, "
            "and n_adults together."
        ),
    }])


def main() -> None:
    logger.info("Reading panel: %s", INPUT_PATH)
    panel = pl.read_parquet(INPUT_PATH)
    panel_cols = set(panel.columns)

    outcomes = [o for o in ANALYSIS_OUTCOMES if o in panel_cols]
    controls = [c for c in BALANCE_CONTROLS if c in panel_cols]

    if not outcomes:
        raise ValueError("No ANALYSIS_OUTCOMES found in panel.")

    logger.info("Primary exposure: %s", PRIMARY_SPEC)
    logger.info("Outcomes available: %s", outcomes)
    logger.info("Controls available: %s", controls)

    # Start with the first outcome. You can change this manually if needed.
    outcome = outcomes[0]
    logger.info("Running diagnostic for outcome: %s", outcome)

    df_event = build_event_study_data(panel, exposure=PRIMARY_SPEC)

    diagnostic = diagnose_controls(
        df_event=df_event,
        outcome=outcome,
        controls=controls,
    )

    diagnostic_path = OUTPUT_DIR / "event_study_control_collinearity_diagnostic.csv"
    diagnostic.to_csv(diagnostic_path, index=False)
    logger.info("Saved: %s", diagnostic_path)

    mechanical = check_mechanical_collinearity(panel)
    mechanical_path = OUTPUT_DIR / "mechanical_collinearity_checks.csv"
    mechanical.to_csv(mechanical_path, index=False)
    logger.info("Saved: %s", mechanical_path)

    logger.info("=== Diagnostic complete ===")


if __name__ == "__main__":
    main()
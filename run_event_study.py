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
    cols = [c for c in ["pretrend_chi2_stat", "pretrend_chi2_p", "pretrend_group_chi2_stat", "pretrend_group_chi2_p", "pretrend_inference", "n_obs", "n_clusters"] if c in df.columns]
    out = df.groupby(group_cols, dropna=False)[cols].first().reset_index()
    out["exposure_label"] = out["exposure_spec"].map(EXPOSURE_LABELS).fillna(out["exposure_spec"])
    out["interpretation"] = out["pretrend_chi2_p"].apply(lambda p: "Diagnostic unavailable" if pd.isna(p) else ("No evidence of differential pre-trends at the 10% level" if p > 0.10 else "Evidence inconsistent with parallel pre-trends at the 10% level"))
    return out


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _print_results(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        return
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    keys = ["exposure_spec", "outcome"]
    if "group" in df.columns and df["group"].notna().any():
        keys.append("group")
    for group_key, block in df.groupby(keys, dropna=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        exposure, outcome = group_key[0], group_key[1]
        group = group_key[2] if len(group_key) > 2 else None
        suffix = f" | group={group}" if group is not None and not pd.isna(group) else ""
        print(f"\n{EXPOSURE_LABELS.get(exposure, exposure)} | outcome={outcome}{suffix}")
        print(f"  {'Year':>4} {'Coef.':>10} {'SE':>10} {'95% CI':>25} {'p_CRV1':>10}")
        for _, row in block.sort_values("year").iterrows():
            ci = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
            print(f"  {int(row['year']):>4} {row['coef']:+10.4f} {row['se']:10.4f} {ci:>25} {row['pval_crv1']:10.4f}{_stars(row['pval_crv1'])}")
        first = block.iloc[0]
        if "pretrend_group_chi2_p" in block.columns and group is not None and not pd.isna(group):
            print(f"  Group pre-trend: chi2={first['pretrend_group_chi2_stat']:.3f}, p={first['pretrend_group_chi2_p']:.4f}")
        print(f"  Joint pre-trend: chi2={first['pretrend_chi2_stat']:.3f}, p={first['pretrend_chi2_p']:.4f}")
        print("  Inference: CRV1 asymptotic chi-square Wald diagnostic")


def _print_summary(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        return
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for _, row in df.iterrows():
        group = row.get("group")
        suffix = "" if pd.isna(group) else f" | group={group}"
        print(f"{row['exposure_label']} | {row['outcome']}{suffix} | chi2={row['pretrend_chi2_stat']:.3f} | p={row['pretrend_chi2_p']:.4f} | {row['interpretation']}")


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
    tercile_summary = _diagnostic_summary(tercile)
    continuous_summary = _diagnostic_summary(continuous)

    tercile.to_csv(OUTPUT_DIR / "event_study_tercile_baseline.csv", index=False)
    continuous.to_csv(OUTPUT_DIR / "event_study_continuous_complementary.csv", index=False)
    groups.to_csv(OUTPUT_DIR / "event_study_tercile_assignments.csv", index=False)
    tercile_summary.to_csv(OUTPUT_DIR / "event_study_tercile_pretrend_summary.csv", index=False)
    continuous_summary.to_csv(OUTPUT_DIR / "event_study_continuous_pretrend_summary.csv", index=False)

    _print_results(tercile, "BASELINE TERCILE EVENT-STUDY COEFFICIENTS")
    _print_summary(tercile_summary, "BASELINE TERCILE PRE-TREND DIAGNOSTICS")
    _print_results(continuous, "COMPLEMENTARY CONTINUOUS EVENT-STUDY COEFFICIENTS")
    _print_summary(continuous_summary, "COMPLEMENTARY CONTINUOUS PRE-TREND DIAGNOSTICS")
    logger.info("Event-study outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

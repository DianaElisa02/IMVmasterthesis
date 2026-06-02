"""
run_event_study.py
Runs three analyses:
  1. Event study         — full sample, all event-study years, all exposure specs
  2. Placebo test        — pre-reform years only (2017–2019), fake post=2019,
                           primary spec only (identification check)
  3. Region trends       — event study + drgn2 x year_centered interactions
                           (robustness check, documented as underpowered),
                           all exposure specs

Outputs
-------
  event_study_all_results.csv   — one merged table, all specs x outcomes x models
  placebo_all_results.csv       — one merged placebo table
  event_study_grid.png          — combined plot: rows=outcomes, cols=exposure specs
  region_trends_grid.png        — same layout for region-trends robustness
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl

from src.event_study import (
    build_event_study_data,
    run_event_study,
    run_placebo,
)
from src.constants import (
    ANALYSIS_OUTCOMES,
    BALANCE_CONTROLS,
    EXPOSURE_SPECS,
    EVENT_STUDY_REFERENCE_YEAR,
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

PRIMARY_SPEC = EXPOSURE_SPECS[0]
_REF_YEAR    = EVENT_STUDY_REFERENCE_YEAR

# Short labels for plot titles
OUTCOME_LABELS = {
    "vhMATDEP":       "Severe material deprivation",
    "vhPobreza":      "At-risk-of-poverty",
    "poverty_gap":    "Poverty gap (FGT-1)",
    "poverty_gap_sq": "Poverty gap sq. (FGT-2)",
}

SPEC_LABELS = {
    spec: (f"{spec} [primary]" if spec == PRIMARY_SPEC else spec)
    for spec in EXPOSURE_SPECS
}


# ---------------------------------------------------------------------------
# Console printing helpers
# ---------------------------------------------------------------------------

def print_event_summary(coef_df: pd.DataFrame, outcome: str) -> None:
    sep   = "-" * 65
    model = coef_df["model"].iloc[0] if "model" in coef_df.columns else ""
    print(f"\n  Outcome: {outcome}  |  Model: {model}")
    print(f"  {sep}")
    print(f"  {'Year':>6}  {'Rel.yr':>6}  {'Coef':>8}  {'SE':>7}  "
          f"{'p_CRV1':>8}  {'p_WCB':>8}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}")

    for _, row in coef_df.sort_values("year").iterrows():
        p_wbt_val = row.get("pval_wbt", float("nan"))
        stars     = ""
        if not pd.isna(p_wbt_val):
            stars = "***" if p_wbt_val < 0.01 else "**" if p_wbt_val < 0.05 else "*" if p_wbt_val < 0.10 else ""
        pval_crv1 = f"{row['pval_crv1']:.4f}" if not pd.isna(row["pval_crv1"]) else "  ref "
        pval_wbt  = f"{p_wbt_val:.4f} {stars}" if not pd.isna(p_wbt_val) else "  ref "
        ref_tag   = " ← ref" if row["rel_year"] == 0 else ""
        print(f"  {int(row['year']):>6}  {int(row['rel_year']):>+6}  "
              f"{row['coef']:>+8.4f}  {row['se']:>7.4f}  "
              f"{pval_crv1:>8}  {pval_wbt:>10}{ref_tag}")

    if "pretrend_wald_p" in coef_df.columns:
        wald_rows = coef_df["pretrend_wald_p"].dropna()
        if not wald_rows.empty:
            wp      = wald_rows.iloc[0]
            ws      = coef_df["pretrend_wald_stat"].dropna().iloc[0]
            verdict = "✓ pre-trends not rejected (p > 0.10)" if wp > 0.10 else "⚠ pre-trends rejected (p ≤ 0.10)"
            print(f"\n  Pre-trend joint Wald: stat={ws:.3f}  p={wp:.4f}  {verdict}")

    if "lead_mean" in coef_df.columns:
        lead_rows = coef_df["lead_mean"].dropna()
        if not lead_rows.empty:
            lm  = lead_rows.iloc[0]
            lma = coef_df["lead_max_abs"].dropna().iloc[0]
            print(f"  Lead diagnostics:     mean={lm:+.4f}  max_abs={lma:.4f}")


def print_placebo_summary(results: list[dict]) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  PLACEBO TEST — fake post = 2019, pre-reform years 2017–2019")
    print("  H0: beta_placebo = 0  (WCB primary inference)")
    print(sep)
    print(f"  {'Outcome':<12}  {'Coef':>8}  {'SE':>7}  {'p_CRV1':>8}  {'p_WCB':>8}  Verdict")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*20}")
    for r in results:
        p       = r["pval_wbt"]
        stars   = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        verdict = "✓ null not rejected" if p > 0.10 else "⚠ significant"
        print(f"  {r['outcome']:<12}  {r['coef']:>+8.4f}  {r['se']:>7.4f}  "
              f"{r['pval_crv1']:>8.4f}  {r['pval_wbt']:>7.4f} {stars:<3}  {verdict}")


# ---------------------------------------------------------------------------
# Combined grid plot
# ---------------------------------------------------------------------------

def _draw_panel(
    ax: plt.Axes,
    coef_df: pd.DataFrame,
    title: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    """Draw one event-study panel onto an existing Axes object."""
    df   = coef_df.sort_values("year").copy()
    pre  = df[df["rel_year"] <  0]
    ref  = df[df["rel_year"] == 0]
    post = df[df["rel_year"] >  0]

    for subset, color, label in [
        (pre,  "#378ADD", "Pre-reform"),
        (post, "#F4A261", "Post-reform"),
    ]:
        if subset.empty:
            continue
        ax.errorbar(
            subset["rel_year"], subset["coef"],
            yerr=subset["coef"] - subset["ci_low"],
            fmt="o", color=color, capsize=3, linewidth=1.2,
            markersize=4, label=label,
        )

    ax.scatter(ref["rel_year"], ref["coef"], color="#2A9D8F",
               zorder=5, s=50, label=f"Ref ({_REF_YEAR})")

    ax.axhline(0,    color="#B4B2A9", linewidth=0.7, linestyle="--")
    ax.axvline(-0.5, color="#B4B2A9", linewidth=0.7, linestyle=":")

    # Wald + lead annotation in top-left corner
    if "pretrend_wald_p" in df.columns:
        wald_rows = df["pretrend_wald_p"].dropna()
        if not wald_rows.empty:
            wp  = wald_rows.iloc[0]
            lm  = df["lead_mean"].dropna().iloc[0] if "lead_mean" in df.columns else np.nan
            ann = f"Wald p={wp:.3f}"
            if not np.isnan(lm):
                ann += f"\nlead μ={lm:+.3f}"
            ax.text(
                0.03, 0.97, ann,
                transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
            )

    ax.set_title(title, fontsize=7.5, pad=3)
    if show_xlabel:
        ax.set_xlabel("Years relative to reform", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("Coef (pp / SD exposure)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, alpha=0.25, linewidth=0.4)


def plot_combined_grid(
    results: dict[tuple[str, str], pd.DataFrame],
    outcomes: list[str],
    exposures: list[str],
    output_path: Path,
    suptitle: str = "Event study",
) -> None:
    """
    Build a grid figure: rows = outcomes, cols = exposure specs.

    Parameters
    ----------
    results     : dict keyed by (exposure, outcome) -> coef_df
    outcomes    : ordered list of outcome names (rows)
    exposures   : ordered list of exposure spec names (cols)
    output_path : where to save the .png
    suptitle    : figure-level title
    """
    n_rows = len(outcomes)
    n_cols = len(exposures)

    fig = plt.figure(figsize=(4.5 * n_cols, 3.5 * n_rows))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
    )

    last_ax = None
    for r, outcome in enumerate(outcomes):
        for c, exposure in enumerate(exposures):
            ax  = fig.add_subplot(gs[r, c])
            key = (exposure, outcome)

            if key not in results:
                ax.text(0.5, 0.5, "no result", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
                ax.set_title(
                    f"{OUTCOME_LABELS.get(outcome, outcome)}\n"
                    f"{SPEC_LABELS.get(exposure, exposure)}",
                    fontsize=7.5,
                )
                continue

            col_label = SPEC_LABELS.get(exposure, exposure)
            row_label = OUTCOME_LABELS.get(outcome, outcome)

            _draw_panel(
                ax,
                results[key],
                title=f"{row_label}\n{col_label}",
                show_xlabel=(r == n_rows - 1),
                show_ylabel=(c == 0),
            )
            last_ax = ax

    # Shared legend below the grid
    if last_ax is not None:
        handles, labels = last_ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc="lower center",
                ncol=3,
                fontsize=8,
                frameon=True,
                bbox_to_anchor=(0.5, -0.02),
            )

    fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Combined grid saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== IMV DiD — run_event_study.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))
    logger.info("Outcomes: %s", ANALYSIS_OUTCOMES)
    logger.info("Exposure specs: %s", EXPOSURE_SPECS)

    controls = [c for c in BALANCE_CONTROLS if c in panel.columns]

    # Accumulators
    all_coef_rows:  list[pd.DataFrame]              = []
    event_results:  dict[tuple[str, str], pd.DataFrame] = {}
    trends_results: dict[tuple[str, str], pd.DataFrame] = {}
    placebo_results: list[dict]                     = []

    # -----------------------------------------------------------------------
    # 1. EVENT STUDY — all exposure specs x all outcomes
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  EVENT STUDY")
    print("  Inference: WCB (primary) | CRV1 (auxiliary)")
    print("=" * 65)

    for exposure in EXPOSURE_SPECS:
        label = "PRIMARY" if exposure == PRIMARY_SPEC else "robustness"
        print(f"\n{'=' * 65}")
        print(f"  Exposure spec: {exposure}  [{label}]")
        print(f"{'=' * 65}")

        df_event  = build_event_study_data(panel, exposure=exposure)
        model_tag = f"baseline_{exposure}"

        for outcome in ANALYSIS_OUTCOMES:
            logger.info("--- Event study [%s]: %s ---", exposure, outcome)
            coef_df = run_event_study(
                df_event,
                outcome=outcome,
                controls=controls,
                output_dir=OUTPUT_DIR,
                region_trends=False,
                model_tag=model_tag,
            )
            coef_df["exposure_spec"] = exposure
            coef_df["outcome"]       = outcome

            print_event_summary(coef_df, outcome)

            event_results[(exposure, outcome)] = coef_df
            all_coef_rows.append(coef_df)

    # -----------------------------------------------------------------------
    # 2. PLACEBO TEST — primary spec only
    # -----------------------------------------------------------------------
    logger.info("=== Placebo test ===")
    for outcome in ANALYSIS_OUTCOMES:
        logger.info("--- Placebo: %s ---", outcome)
        result = run_placebo(
            panel,
            outcome=outcome,
            controls=controls,
            output_dir=OUTPUT_DIR,
        )
        placebo_results.append(result)

    print_placebo_summary(placebo_results)

    placebo_df   = pd.DataFrame(placebo_results)
    placebo_path = OUTPUT_DIR / "placebo_all_results.csv"
    placebo_df.to_csv(placebo_path, index=False)
    logger.info("Placebo table saved: %s", placebo_path)

    # -----------------------------------------------------------------------
    # 3. REGION-SPECIFIC TIME TRENDS — robustness, all exposure specs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  REGION-SPECIFIC TIME TRENDS (robustness)")
    print("  Note: underpowered with 17 clusters — for documentation only")
    print("=" * 65)

    for exposure in EXPOSURE_SPECS:
        print(f"\n  Exposure spec: {exposure}")

        df_event = build_event_study_data(panel, exposure=exposure)

        for outcome in ANALYSIS_OUTCOMES:
            logger.info("--- Region trends [%s]: %s ---", exposure, outcome)
            try:
                coef_df_rt = run_event_study(
                    df_event,
                    outcome=outcome,
                    controls=controls,
                    output_dir=OUTPUT_DIR,
                    region_trends=True,
                    model_tag="region_trends",
                )
                coef_df_rt["exposure_spec"] = exposure
                coef_df_rt["outcome"]       = outcome

                print_event_summary(coef_df_rt, outcome)

                trends_results[(exposure, outcome)] = coef_df_rt
                all_coef_rows.append(coef_df_rt)

            except Exception as e:
                logger.error("Region trends failed for %s [%s]: %s", outcome, exposure, e)
                print(f"  Region trends failed for {outcome} [{exposure}]: {e}")

    # -----------------------------------------------------------------------
    # Merged coefficient table (all specs x outcomes x models)
    # -----------------------------------------------------------------------
    if all_coef_rows:
        merged      = pd.concat(all_coef_rows, ignore_index=True)
        merged_path = OUTPUT_DIR / "event_study_all_results.csv"
        merged.to_csv(merged_path, index=False)
        logger.info("Merged results table saved: %s", merged_path)

    # -----------------------------------------------------------------------
    # Combined grid plots
    # -----------------------------------------------------------------------
    if event_results:
        plot_combined_grid(
            results=event_results,
            outcomes=ANALYSIS_OUTCOMES,
            exposures=EXPOSURE_SPECS,
            output_path=OUTPUT_DIR / "event_study_grid.png",
            suptitle=(
                "Event study — all outcomes × exposure specs\n"
                "Coefficient: pp change per SD exposure  |  95% CI (CRV1)  |  Reference: 2019"
            ),
        )

    if trends_results:
        plot_combined_grid(
            results=trends_results,
            outcomes=ANALYSIS_OUTCOMES,
            exposures=EXPOSURE_SPECS,
            output_path=OUTPUT_DIR / "region_trends_grid.png",
            suptitle=(
                "Region-specific time trends (robustness) — all outcomes × exposure specs\n"
                "Note: underpowered with 17 clusters — do not report as primary evidence"
            ),
        )

    logger.info("Event study complete. Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
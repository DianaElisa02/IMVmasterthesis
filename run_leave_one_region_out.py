"""Leave-one-region-out robustness checks for the preferred tercile DiD.

Two variants are estimated for the 2021-2025 baseline post period:

1. fixed_groups
   Full-sample tercile assignments are retained, then one region is removed.
   This isolates the influence of each region on the preferred grouped estimate.

2. reranked_groups
   The omitted region is removed first, then terciles are recomputed among the
   remaining 16 regions. This additionally tests sensitivity to the grouping
   procedure.

The script covers both co-primary hybrid exposure measures and all four analysis
outcomes. Because the purpose is coefficient-influence diagnostics rather than
repeated hypothesis testing, leave-one-out runs use region-clustered standard
errors and t(G-1) confidence intervals without wild-cluster bootstrap p-values.
The canonical full-sample wild-bootstrap inference remains in run_binned_did.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
from scipy.stats import t as t_dist

from src.binned_did import compute_tercile_assignments
from src.constants import ANALYSIS_OUTCOMES, DID_POST_YEARS_BASELINE, REGION_NAMES, YEARS
from src.control_specs import PREFERRED_CONTROLS, add_preferred_control_groups, cast_categorical_controls
from src.exposure_specs import EXPOSURE_LABELS, PRIMARY_EXPOSURE_SPECS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "robustness" / "leave_one_region_out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_LABELS = {"medium": "Medium exposure", "high": "High exposure"}
OUTCOME_LABELS = {
    "poverty": "At-risk-of-poverty",
    "matdep": "Severe material deprivation",
    "poverty_gap": "Poverty gap",
    "poverty_gap_sq": "Squared poverty gap",
}


def _build_did_from_assignments(
    panel: pl.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    did = (
        panel.filter(pl.col("year").is_in(YEARS + DID_POST_YEARS_BASELINE))
        .with_columns(
            pl.col("year").is_in(DID_POST_YEARS_BASELINE).cast(pl.Float64).alias("post")
        )
        .join(
            pl.from_pandas(assignments[["drgn2", "exposure_tercile"]]),
            on="drgn2",
            how="inner",
        )
        .with_columns(
            pl.col("exposure_tercile").eq("medium").cast(pl.Float64).alias("tercile_medium"),
            pl.col("exposure_tercile").eq("high").cast(pl.Float64).alias("tercile_high"),
        )
        .with_columns(
            (pl.col("post") * pl.col("tercile_medium")).alias("post_x_medium"),
            (pl.col("post") * pl.col("tercile_high")).alias("post_x_high"),
        )
    )
    return did.to_pandas()


def _estimate(df: pd.DataFrame, outcome: str) -> dict:
    required = [
        outcome,
        "post_x_medium",
        "post_x_high",
        "drgn2",
        "year",
        *PREFERRED_CONTROLS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing estimation columns: {missing}")

    clean = cast_categorical_controls(df[required].dropna().reset_index(drop=True))
    formula = (
        f"{outcome} ~ post_x_medium + post_x_high + "
        f"{' + '.join(PREFERRED_CONTROLS)} | drgn2 + year"
    )
    fit = pf.feols(formula, data=clean, vcov={"CRV1": "drgn2"})
    n_clusters = int(clean["drgn2"].nunique())
    crit = float(t_dist.ppf(0.975, df=n_clusters - 1))

    result = {
        "outcome": outcome,
        "n_obs": int(len(clean)),
        "n_clusters": n_clusters,
        "controls": ";".join(PREFERRED_CONTROLS),
        "control_spec": "preferred_demographic",
    }
    for group in ["medium", "high"]:
        term = f"post_x_{group}"
        coef = float(fit.coef()[term])
        se = float(fit.se()[term])
        result.update(
            {
                f"coef_{group}": coef,
                f"se_{group}": se,
                f"ci_low_{group}": coef - crit * se,
                f"ci_high_{group}": coef + crit * se,
                f"pval_cluster_{group}": float(fit.pvalue()[term]),
            }
        )
    return result


def _full_sample_rows(panel: pl.DataFrame, exposure_spec: str) -> list[dict]:
    assignments = compute_tercile_assignments(panel, exposure_spec)
    df = _build_did_from_assignments(panel, assignments)
    rows: list[dict] = []
    for outcome in ANALYSIS_OUTCOMES:
        result = _estimate(df, outcome)
        result.update(
            {
                "exposure_spec": exposure_spec,
                "loo_variant": "full_sample",
                "omitted_region": np.nan,
                "omitted_region_name": "Full sample",
                "omitted_original_tercile": np.nan,
                "n_assignment_changes": 0,
            }
        )
        rows.append(result)
    return rows


def _loo_rows(panel: pl.DataFrame, exposure_spec: str) -> list[dict]:
    full_assignments = compute_tercile_assignments(panel, exposure_spec)
    full_map = full_assignments.set_index("drgn2")["exposure_tercile"].to_dict()
    rows: list[dict] = []

    for omitted_region in sorted(full_assignments["drgn2"].astype(int).tolist()):
        omitted_name = REGION_NAMES.get(omitted_region, str(omitted_region))
        reduced_panel = panel.filter(pl.col("drgn2") != omitted_region)

        fixed_assignments = full_assignments[
            full_assignments["drgn2"].ne(omitted_region)
        ][["drgn2", "exposure_tercile"]].copy()

        reranked_assignments = compute_tercile_assignments(
            reduced_panel,
            exposure_spec,
        )[["drgn2", "exposure_tercile"]].copy()

        reranked_map = reranked_assignments.set_index("drgn2")["exposure_tercile"].to_dict()
        assignment_changes = sum(
            reranked_map[region] != full_map[region]
            for region in reranked_map
        )

        for variant, assignments, n_changes in [
            ("fixed_groups", fixed_assignments, 0),
            ("reranked_groups", reranked_assignments, assignment_changes),
        ]:
            df = _build_did_from_assignments(reduced_panel, assignments)
            for outcome in ANALYSIS_OUTCOMES:
                result = _estimate(df, outcome)
                result.update(
                    {
                        "exposure_spec": exposure_spec,
                        "loo_variant": variant,
                        "omitted_region": omitted_region,
                        "omitted_region_name": omitted_name,
                        "omitted_original_tercile": full_map[omitted_region],
                        "n_assignment_changes": int(n_changes),
                    }
                )
                rows.append(result)

        logger.info(
            "%s | omitted %s | reranked assignment changes=%d",
            exposure_spec,
            omitted_name,
            assignment_changes,
        )

    return rows


def _make_long(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    id_columns = [
        "exposure_spec",
        "loo_variant",
        "omitted_region",
        "omitted_region_name",
        "omitted_original_tercile",
        "n_assignment_changes",
        "outcome",
        "n_obs",
        "n_clusters",
        "controls",
        "control_spec",
    ]
    for _, row in results.iterrows():
        base = {column: row[column] for column in id_columns}
        for group in ["medium", "high"]:
            rows.append(
                {
                    **base,
                    "group": group,
                    "coef": row[f"coef_{group}"],
                    "se": row[f"se_{group}"],
                    "ci_low": row[f"ci_low_{group}"],
                    "ci_high": row[f"ci_high_{group}"],
                    "pval_cluster": row[f"pval_cluster_{group}"],
                }
            )
    return pd.DataFrame(rows)


def _summarise(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = (
        long[long["loo_variant"].eq("full_sample")]
        .set_index(["exposure_spec", "outcome", "group"])["coef"]
    )
    loo = long[long["loo_variant"].ne("full_sample")].copy()
    loo["full_sample_coef"] = [
        full.loc[(spec, outcome, group)]
        for spec, outcome, group in zip(
            loo["exposure_spec"], loo["outcome"], loo["group"]
        )
    ]
    loo["absolute_change"] = (loo["coef"] - loo["full_sample_coef"]).abs()
    loo["sign_change"] = np.sign(loo["coef"]) != np.sign(loo["full_sample_coef"])

    summary = (
        loo.groupby(["exposure_spec", "loo_variant", "outcome", "group"], observed=True)
        .agg(
            full_sample_coef=("full_sample_coef", "first"),
            min_loo_coef=("coef", "min"),
            max_loo_coef=("coef", "max"),
            max_absolute_change=("absolute_change", "max"),
            n_sign_changes=("sign_change", "sum"),
            most_influential_region=(
                "omitted_region_name",
                lambda x: x.iloc[loo.loc[x.index, "absolute_change"].argmax()],
            ),
            max_assignment_changes=("n_assignment_changes", "max"),
        )
        .reset_index()
    )
    return loo, summary


def _plot(long_with_deltas: pd.DataFrame, exposure_spec: str, variant: str) -> None:
    block = long_with_deltas[
        long_with_deltas["exposure_spec"].eq(exposure_spec)
        & long_with_deltas["loo_variant"].eq(variant)
    ].copy()

    for outcome in ANALYSIS_OUTCOMES:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.8), sharey=True)
        outcome_block = block[block["outcome"].eq(outcome)]

        for ax, group in zip(axes, ["medium", "high"]):
            series = outcome_block[outcome_block["group"].eq(group)].copy()
            if series.empty:
                raise ValueError(
                    f"No leave-one-out rows for {exposure_spec}, {variant}, "
                    f"{outcome}, {group}"
                )
            series = series.sort_values("coef")
            full_sample_coef = float(series["full_sample_coef"].iloc[0])
            y = np.arange(len(series))
            ax.errorbar(
                series["coef"],
                y,
                xerr=[series["coef"] - series["ci_low"], series["ci_high"] - series["coef"]],
                fmt="o",
                capsize=2.5,
            )
            ax.axvline(full_sample_coef, linestyle="--", linewidth=1.2)
            ax.axvline(0, linewidth=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(series["omitted_region_name"])
            ax.set_title(GROUP_LABELS[group])
            ax.set_xlabel("DiD coefficient")
            ax.grid(axis="x", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            f"Leave-one-region-out: {EXPOSURE_LABELS[exposure_spec]} — "
            f"{OUTCOME_LABELS[outcome]} ({variant.replace('_', ' ')})"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = OUTPUT_DIR / f"fig_loo_{variant}_{exposure_spec}_{outcome}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure: %s", path)


def main() -> None:
    panel = add_preferred_control_groups(pl.read_parquet(INPUT_PATH))
    rows: list[dict] = []

    for exposure_spec in PRIMARY_EXPOSURE_SPECS:
        rows.extend(_full_sample_rows(panel, exposure_spec))
        rows.extend(_loo_rows(panel, exposure_spec))

    wide = pd.DataFrame(rows)
    long = _make_long(wide)
    long_with_deltas, summary = _summarise(long)

    wide.to_csv(OUTPUT_DIR / "leave_one_region_out_wide.csv", index=False)
    long_with_deltas.to_csv(OUTPUT_DIR / "leave_one_region_out_long.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "leave_one_region_out_summary.csv", index=False)

    for exposure_spec in PRIMARY_EXPOSURE_SPECS:
        for variant in ["fixed_groups", "reranked_groups"]:
            _plot(long_with_deltas, exposure_spec, variant)

    print("\nLEAVE-ONE-REGION-OUT SUMMARY")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    logger.info("Leave-one-region-out outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

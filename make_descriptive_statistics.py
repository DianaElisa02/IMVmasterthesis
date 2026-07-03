"""Create descriptive statistics for the preferred grouped-DiD estimation sample.

The script mirrors the preferred baseline sample construction by:

1. loading ``output/analysis_dataset_with_gap.parquet``;
2. adding the grouped household-composition controls used in the regressions;
3. restricting to the baseline years used by the preferred grouped DiD;
4. keeping observations complete on all preferred controls and analysis outcomes;
5. calculating weighted outcome statistics and weighted shares for every
   category of the preferred demographic controls.

By default, the script expects the ECV household cross-sectional weight to be
present in the analysis parquet under one of the recognised names. If it is not
present, an external CSV or parquet file can be supplied with ``--weights``.
The external file must share household and year identifiers with the analysis
parquet.

Outputs
-------
``output/descriptive_statistics/descriptive_statistics.csv``
``output/descriptive_statistics/descriptive_statistics.tex``
``output/descriptive_statistics/sample_audit.csv``

Examples
--------
python make_descriptive_statistics.py
python make_descriptive_statistics.py --weights path/to/household_weights.parquet
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.constants import ANALYSIS_OUTCOMES, DID_POST_YEARS_BASELINE, YEARS
from src.control_specs import PREFERRED_CONTROLS, add_preferred_control_groups


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "descriptive_statistics"

WEIGHT_CANDIDATES = ["dwt", "DB090", "weight", "survey_weight"]
HOUSEHOLD_ID_CANDIDATES = ["idhh", "IDHH", "household_id", "hh_id"]
INCOME_CANDIDATES = [
    "net_annual_household_income",
    "net_household_income",
    "household_income",
    "hy020",
    "yds",
]

OUTCOME_LABELS = {
    "poverty": "At-risk-of-poverty",
    "matdep": "Severe material deprivation",
    "poverty_gap": "Poverty gap",
    "poverty_gap_sq": "Squared poverty gap",
}

CATEGORY_LABELS = {
    "head_age_group": {
        "under35": "Household head aged under 35",
        "35_54": "Household head aged 35--54",
        "55_64": "Household head aged 55--64",
        "65plus": "Household head aged 65 or older",
    },
    "head_sex": {
        "female": "Female household head",
        "male": "Male household head",
    },
    "head_education_group": {
        "low": "Low education",
        "medium": "Medium education",
        "high": "High education",
    },
    "n_adults_group": {
        "1": "One adult",
        "2": "Two adults",
        "3plus": "Three or more adults",
    },
    "n_children_group": {
        "0": "No children",
        "1": "One child",
        "2": "Two children",
        "3plus": "Three or more children",
    },
}

CATEGORY_ORDER = {
    "head_age_group": ["under35", "35_54", "55_64", "65plus"],
    "head_sex": ["female", "male"],
    "head_education_group": ["low", "medium", "high"],
    "n_adults_group": ["1", "2", "3plus"],
    "n_children_group": ["0", "1", "2", "3plus"],
}

PANEL_LABELS = {
    "outcomes": "Panel A: Outcome variables",
    "head": "Panel B: Household-head characteristics",
    "composition": "Panel C: Household composition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional CSV/parquet file containing the household survey weight.",
    )
    parser.add_argument(
        "--weight-column",
        default=None,
        help="Weight-column name. If omitted, recognised names are detected automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    return parser.parse_args()


def read_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=10000)
    raise ValueError(f"Unsupported file type: {path}")


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def attach_weight(
    panel: pl.DataFrame,
    weights_path: Path | None,
    requested_weight: str | None,
) -> tuple[pl.DataFrame, str]:
    weight_col = requested_weight or detect_column(panel.columns, WEIGHT_CANDIDATES)
    if weight_col and weight_col in panel.columns:
        return panel, weight_col

    if weights_path is None:
        raise ValueError(
            "No ECV household weight was found in the analysis parquet. "
            "Retain 'dwt' when constructing the analysis dataset or rerun this script "
            "with --weights pointing to a CSV/parquet file containing household-year weights."
        )

    weights = read_table(weights_path)
    weight_col = requested_weight or detect_column(weights.columns, WEIGHT_CANDIDATES)
    if weight_col is None:
        raise ValueError(
            f"No recognised weight column found in {weights_path}. "
            f"Expected one of {WEIGHT_CANDIDATES}, or pass --weight-column."
        )
    if weight_col not in weights.columns:
        raise ValueError(f"Requested weight column '{weight_col}' is absent from {weights_path}")

    shared_year = "year" if "year" in panel.columns and "year" in weights.columns else None
    shared_id = next(
        (
            name
            for name in HOUSEHOLD_ID_CANDIDATES
            if name in panel.columns and name in weights.columns
        ),
        None,
    )
    if shared_year is None or shared_id is None:
        raise ValueError(
            "The external weights file must share 'year' and a household identifier "
            f"with the analysis parquet. Recognised identifiers: {HOUSEHOLD_ID_CANDIDATES}."
        )

    keys = [shared_year, shared_id]
    weights = weights.select(keys + [weight_col]).unique(subset=keys, keep="first")
    panel = panel.join(weights, on=keys, how="left", validate="m:1")
    return panel, weight_col


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return math.nan
    x = pd.to_numeric(values[mask], errors="coerce")
    w = pd.to_numeric(weights[mask], errors="coerce")
    valid = x.notna() & w.notna() & np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return math.nan
    return float(np.average(x[valid], weights=w[valid]))


def weighted_sd(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return math.nan
    x = pd.to_numeric(values[mask], errors="coerce")
    w = pd.to_numeric(weights[mask], errors="coerce")
    valid = x.notna() & w.notna() & np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[valid].to_numpy(dtype=float)
    w = w[valid].to_numpy(dtype=float)
    if len(x) == 0 or w.sum() <= 0:
        return math.nan
    mean = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mean) ** 2, weights=w)))


def build_exact_sample(panel: pl.DataFrame, weight_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = add_preferred_control_groups(panel)
    baseline_years = YEARS + DID_POST_YEARS_BASELINE

    required = [
        "year",
        "drgn2",
        weight_col,
        *ANALYSIS_OUTCOMES,
        *PREFERRED_CONTROLS,
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"Required variables missing from analysis data: {missing}")

    restricted = panel.filter(pl.col("year").is_in(baseline_years))
    n_before = restricted.height

    valid_weight = (
        pl.col(weight_col).cast(pl.Float64, strict=False).is_not_null()
        & pl.col(weight_col).cast(pl.Float64, strict=False).is_finite()
        & (pl.col(weight_col).cast(pl.Float64, strict=False) > 0)
    )
    complete_expr = pl.all_horizontal([pl.col(column).is_not_null() for column in required])
    sample = restricted.filter(complete_expr & valid_weight)

    audit = pd.DataFrame(
        [
            {"stage": "Analysis parquet", "observations": panel.height},
            {"stage": "Baseline years", "observations": n_before},
            {
                "stage": "Complete preferred estimation sample with positive weight",
                "observations": sample.height,
            },
        ]
    )

    return sample.to_pandas(), audit


def continuous_row(
    df: pd.DataFrame,
    variable: str,
    label: str,
    panel: str,
    weight_col: str,
) -> dict:
    values = pd.to_numeric(df[variable], errors="coerce")
    return {
        "panel": panel,
        "variable": variable,
        "label": label,
        "statistic_type": "continuous_or_binary_outcome",
        "mean": weighted_mean(values, df[weight_col]),
        "std_dev": weighted_sd(values, df[weight_col]),
        "observations": int(values.notna().sum()),
    }


def category_rows(
    df: pd.DataFrame,
    variable: str,
    panel: str,
    weight_col: str,
) -> list[dict]:
    rows: list[dict] = []
    series = df[variable].astype("string")
    for value in CATEGORY_ORDER[variable]:
        indicator = series.eq(value).astype(float)
        rows.append(
            {
                "panel": panel,
                "variable": variable,
                "category": value,
                "label": CATEGORY_LABELS[variable][value],
                "statistic_type": "category_share",
                "mean": weighted_mean(indicator, df[weight_col]),
                "std_dev": weighted_sd(indicator, df[weight_col]),
                "observations": int(series.notna().sum()),
            }
        )
    return rows


def build_statistics(df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    rows: list[dict] = []

    for outcome in ANALYSIS_OUTCOMES:
        rows.append(
            continuous_row(
                df,
                outcome,
                OUTCOME_LABELS.get(outcome, outcome),
                "outcomes",
                weight_col,
            )
        )

    income_var = detect_column(list(df.columns), INCOME_CANDIDATES)
    if income_var:
        rows.append(
            continuous_row(
                df,
                income_var,
                "Net annual household income (\\euro)",
                "outcomes",
                weight_col,
            )
        )

    for variable in ["head_age_group", "head_sex", "head_education_group"]:
        rows.extend(category_rows(df, variable, "head", weight_col))

    for variable in ["n_adults_group", "n_children_group"]:
        rows.extend(category_rows(df, variable, "composition", weight_col))

    return pd.DataFrame(rows)


def latex_escape_label(label: str) -> str:
    return label.replace("%", r"\%").replace("&", r"\&")


def format_number(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{decimals}f}"


def write_latex(stats: pd.DataFrame, output_path: Path, weight_col: str, n_obs: int) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Descriptive statistics for the preferred estimation sample}",
        r"\label{tab:descriptive_statistics}",
        r"\small",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variable & Mean & Std. dev. & Observations \\",
        r"\midrule",
    ]

    panel_order = ["outcomes", "head", "composition"]
    for panel_index, panel in enumerate(panel_order):
        if panel_index > 0:
            lines.append(r"\addlinespace[0.5em]")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{PANEL_LABELS[panel]}}}}} \\")
        lines.append(r"\addlinespace[0.2em]")
        block = stats.loc[stats["panel"].eq(panel)]
        for row in block.itertuples(index=False):
            label = latex_escape_label(row.label)
            mean = format_number(row.mean)
            sd = format_number(row.std_dev)
            observations = f"{int(row.observations):,}"
            lines.append(f"{label} & {mean} & {sd} & {observations} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            (
                r"\item \textit{Notes:} The table reports weighted descriptive statistics "
                r"for the pooled preferred grouped difference-in-differences estimation sample. "
                r"The sample covers the 2017--2019 and 2021--2025 survey waves. "
                rf"Statistics are calculated using the ECV cross-sectional household weight "
                rf"\texttt{{{weight_col}}}. Means for indicator variables and category rows are "
                rf"weighted population shares. Observations are household-year records. "
                rf"The common complete-case sample contains {n_obs:,} observations."
            ),
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = read_table(args.input)
    panel, weight_col = attach_weight(panel, args.weights, args.weight_column)
    sample, audit = build_exact_sample(panel, weight_col)

    stats = build_statistics(sample, weight_col)
    stats.to_csv(args.output_dir / "descriptive_statistics.csv", index=False)
    audit.to_csv(args.output_dir / "sample_audit.csv", index=False)
    write_latex(
        stats,
        args.output_dir / "descriptive_statistics.tex",
        weight_col,
        len(sample),
    )

    print(f"Weight column: {weight_col}")
    print(audit.to_string(index=False))
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()

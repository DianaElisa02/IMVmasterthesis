"""Preferred household controls for the IMV DiD specifications."""
from __future__ import annotations

import pandas as pd
import polars as pl

PREFERRED_CONTROLS: list[str] = [
    "head_age_group",
    "head_sex",
    "head_education_group",
    "n_adults_group",
    "n_children_group",
]

CATEGORICAL_CONTROLS: set[str] = {
    "head_age_group",
    "head_sex",
    "head_education_group",
    "n_adults_group",
    "n_children_group",
    "head_labour_group",
}


def add_preferred_control_groups(panel: pl.DataFrame) -> pl.DataFrame:
    """Add grouped adult and child counts used in the adjusted specifications.

    Adults are grouped as 1, 2, and 3+; children as 0, 1, 2, and 3+.
    Invalid or missing counts remain missing rather than being folded into a
    valid category.
    """
    expressions: list[pl.Expr] = []

    if "n_adults_group" not in panel.columns:
        if "n_adults" not in panel.columns:
            raise ValueError("Cannot construct n_adults_group: n_adults is absent")
        expressions.append(
            pl.when(pl.col("n_adults").eq(1)).then(pl.lit("1"))
            .when(pl.col("n_adults").eq(2)).then(pl.lit("2"))
            .when(pl.col("n_adults").ge(3)).then(pl.lit("3plus"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("n_adults_group")
        )

    if "n_children_group" not in panel.columns:
        if "n_children" not in panel.columns:
            raise ValueError("Cannot construct n_children_group: n_children is absent")
        expressions.append(
            pl.when(pl.col("n_children").eq(0)).then(pl.lit("0"))
            .when(pl.col("n_children").eq(1)).then(pl.lit("1"))
            .when(pl.col("n_children").eq(2)).then(pl.lit("2"))
            .when(pl.col("n_children").ge(3)).then(pl.lit("3plus"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("n_children_group")
        )

    return panel.with_columns(expressions) if expressions else panel


def cast_categorical_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Cast available categorical controls to pandas categorical dtype."""
    out = df.copy()
    for column in CATEGORICAL_CONTROLS:
        if column in out.columns:
            out[column] = out[column].astype("category")
    return out

"""
exposure_io.py

"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exposure_index import PRIMARY_SPECS, SPECS

logger = logging.getLogger(__name__)


def save_exposure(
    exposure_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- exposure_index.csv ---
    spec_cols      = [s["name"] for s in SPECS]
    spec_rank_cols = [f"{s['name']}_rank" for s in SPECS]

    delta_cols = [
        "delta_benefit_hybrid",
        "delta_cov_hybrid",
        "delta_benefit_sim",
        "delta_cov_sim",
        "level_benefit_admin",
        "level_cov_admin",
        "delta_mean",
    ]
    raw_cols = [
        "poor_hh_sim",

        "rmi_exp_sim",
        "post_exp_sim",

        "rmi_rec_sim",
        "post_rec_sim",

        "rmi_mean_sim",
        "imv_mean_sim",

        "rmi_avg_benefit_sim",
        "post_avg_benefit_sim",

        "rmi_coverage_sim",
        "post_coverage_sim",

        "avg_rmi_exp_admin",
        "avg_titulares_admin",
        "rmi_avg_benefit_admin",
        "rmi_coverage_admin",

        "pop",
    ]

    out_cols = (
        ["drgn2", "region"]
        + spec_cols
        + spec_rank_cols
        + [c for c in delta_cols if c in exposure_df.columns]
        + [c for c in raw_cols   if c in exposure_df.columns]
    )
    out_cols = [c for c in out_cols if c in exposure_df.columns]

    exp_path = output_dir / "exposure_index.csv"
    exposure_df[out_cols].to_csv(exp_path, index=False)
    logger.info("Exposure index saved → %s", exp_path)

    std_params = exposure_df.attrs.get("std_params", {})
    if std_params:
        params_rows = [
            {"dimension": dim, "raw_mean": v["raw_mean"], "std": v["std"]}
            for dim, v in std_params.items()
        ]
        pd.DataFrame(params_rows).to_csv(
            output_dir / "exposure_params.csv", index=False
        )
        logger.info(
            "Standardisation parameters saved → %s",
            output_dir / "exposure_params.csv",
        )
def plot_exposure(
    exposure_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot the two co-primary hybrid exposure measures separately.

    Panel A: change in coverage among poor households.
    Panel B: change in average annual benefit among recipient households.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_specs = exposure_df.attrs.get(
        "primary_specs",
        PRIMARY_SPECS,
    )

    if not isinstance(primary_specs, (list, tuple)):
        primary_specs = [primary_specs]

    missing = [
        spec for spec in primary_specs
        if spec not in exposure_df.columns
    ]
    if missing:
        raise KeyError(
            "Cannot plot co-primary exposure measures. "
            f"Missing columns: {missing}"
        )

    if len(primary_specs) != 2:
        raise ValueError(
            "plot_exposure expects exactly two co-primary specifications. "
            f"Received: {primary_specs}"
        )

    coverage_spec = "exposure_cov_hybrid"
    benefit_spec = "exposure_exp_hybrid"

    required = {coverage_spec, benefit_spec, "region"}
    missing_required = required - set(exposure_df.columns)
    if missing_required:
        raise KeyError(
            "Cannot construct exposure plot. "
            f"Missing columns: {sorted(missing_required)}"
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 9),
    )

    plot_settings = [
        {
            "column": coverage_spec,
            "title": "Coverage Exposure",
            "xlabel": (
                "Standardised change in coverage among poor households\n"
                "(positive = higher post-reform coverage)"
            ),
        },
        {
            "column": benefit_spec,
            "title": "Average-Benefit Exposure",
            "xlabel": (
                "Standardised change in average annual benefit\n"
                "(positive = higher post-reform benefit)"
            ),
        },
    ]

    for ax, settings in zip(axes, plot_settings):
        column = settings["column"]

        df_plot = (
            exposure_df[
                ["region", column]
            ]
            .dropna(subset=[column])
            .sort_values(column, ascending=False)
            .reset_index(drop=True)
        )

        values = df_plot[column].to_numpy(dtype=float)
        regions = df_plot["region"].to_numpy()
        
        xmin = np.nanmin(values)
        xmax = np.nanmax(values)
        padding = 0.18 * (xmax - xmin)

        ax.set_xlim(xmin - padding, xmax + padding)
        

        colors = [
            "#378ADD" if value >= 0 else "#E05C5C"
            for value in values
        ]

        bars = ax.barh(
            regions,
            values,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            height=0.72,
        )

        value_range = (
            np.nanmax(values) - np.nanmin(values)
            if len(values) > 0
            else 0
        )
        label_offset = (
            0.02 * value_range
            if value_range > 0
            else 0.03
        )

        for bar, value in zip(bars, values):
            if value >= 0:
                x_position = bar.get_width() + label_offset
                horizontal_alignment = "left"
            else:
                x_position = bar.get_width() - label_offset
                horizontal_alignment = "right"

            ax.text(
                x_position,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha=horizontal_alignment,
                fontsize=8,
            )

        ax.axvline(
            0,
            linewidth=0.8,
            linestyle="--",
        )
        ax.set_xlabel(
            settings["xlabel"],
            fontsize=9,
        )
        ax.set_title(
            settings["title"] + "\n"
            "Pooled 2017–2019, 2022 IMV rules",
            fontsize=10,
            pad=10,
        )
        ax.grid(
            axis="x",
            alpha=0.3,
            linewidth=0.5,
        )
        ax.invert_yaxis()

    fig.suptitle(
        "Regional exposure to the IMV reform: co-primary margins",
        fontsize=13,
        y=1.01,
    )

    plt.tight_layout()

    output_path = output_dir / "exposure_primary_measures.png"
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(
        "Exposure plot saved → %s",
        output_path,
    )
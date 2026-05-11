"""
exposure_io.py
==============
Saving and plotting functions for the exposure variable pipeline.

Responsibilities
----------------
save_exposure()       — writes exposure_index.csv and exposure_params.csv

plot_exposure()       — two-panel figure: composite bar + component comparison
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exposure_index import PRIMARY_SPEC, SPECS

logger = logging.getLogger(__name__)


def save_exposure(
    exposure_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save exposure index and standardisation parameters to CSV.

    exposure_index.csv contains:
      - All five continuous exposure specifications
      - Rank version of each specification
      - Underlying delta dimensions (all variants)
      - Raw averaged simulated and administrative components
      - Descriptive delta_mean

    exposure_params.csv contains:
      - Standardisation mean and std for every input dimension used
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- exposure_index.csv ---
    spec_cols      = [s["name"] for s in SPECS]
    spec_rank_cols = [f"{s['name']}_rank" for s in SPECS]

    delta_cols = [
        "delta_exp_hybrid", "delta_cov_hybrid",
        "delta_exp_sim",    "delta_cov_sim",
        "delta_exp_admin",  "delta_cov_admin",
        "delta_mean",
    ]
    raw_cols = [
        "rmi_exp_sim", "imv_exp_sim",
        "rmi_rec_sim", "imv_rec_sim",
        "rmi_mean_sim", "imv_mean_sim",
        "avg_rmi_exp_admin", "avg_titulares_admin",
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

    # --- exposure_params.csv ---
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
    Two-panel figure:
      Left  — primary specification bar chart (exposure_composite_hybrid)
      Right — all five specifications side by side per region

    Primary spec is read from exposure_df.attrs["primary_spec"] or
    falls back to PRIMARY_SPEC from exposure_index.py.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = exposure_df.attrs.get("primary_spec", PRIMARY_SPEC)
    spec_cols = [s["name"] for s in SPECS]

    # Ensure sorted by primary spec descending
    df = exposure_df.sort_values(primary, ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 8),
        gridspec_kw={"width_ratios": [2, 1.8]}
    )

    # --- Panel 1: primary specification ---
    ax    = axes[0]
    vals  = df[primary].values
    regs  = df["region"].values
    colors = ["#378ADD" if v >= 0 else "#E05C5C" for v in vals]

    bars = ax.barh(regs, vals, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.72)
    for bar, val in zip(bars, vals):
        x_pos = bar.get_width() + (0.03 if val >= 0 else -0.03)
        ha    = "left" if val >= 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha=ha, fontsize=8, color="#3A3A3A",
        )
    ax.axvline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
    ax.set_xlabel(
        f"Exposure score — {primary}\n"
        "(standardised; positive = IMV more generous than pre-reform RMI)",
        fontsize=9,
    )
    ax.set_title(
        f"Primary exposure: {primary}\n"
        "Pooled 2017–2019, 2022 IMV rules (average before differencing)\n"
        "excl. La Rioja, Aragón, Ceuta, Melilla",
        fontsize=9, pad=10,
    )
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.invert_yaxis()

    # --- Panel 2: all specifications ---
    ax2   = axes[1]
    y     = np.arange(len(df))
    n     = len(spec_cols)
    width = 0.72 / n

    colors_spec = [
        "#378ADD", "#F4A261", "#2A9D8F", "#E76F51", "#264653"
    ]
    short_labels = [
        "hybrid", "exp_hyb", "cov_hyb", "sim", "admin"
    ]

    for idx, (col, color, label) in enumerate(
        zip(spec_cols, colors_spec, short_labels)
    ):
        offset = (idx - n / 2 + 0.5) * width
        ax2.barh(
            y + offset, df[col].values,
            height=width, color=color, alpha=0.85,
            label=label, edgecolor="white",
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels(df["region"].values, fontsize=7)
    ax2.axvline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Standardised exposure", fontsize=9)
    ax2.set_title("All specifications\n(for comparison)", fontsize=9, pad=10)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax2.invert_yaxis()

    plt.tight_layout()
    out_path = output_dir / "exposure_index.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Exposure plot saved → %s", out_path)
    plt.close()
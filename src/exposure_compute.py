"""
exposure_compute.py
===================
Computes the household-level gain and regional exposure index.

Gain definition (per household h in region r, year t):
    Gain_h,r,t = Post_protection_h - Pre_protection_h,r,t

Where:
    Post_protection_h  = bsa00_s + bsarg_s_post  (IMV + complementary RMI)
    Pre_protection_h   = bsarg_s_pre             (pre-reform RMI)

For regions where RMI is INCOMPATIBLE with IMV (Galicia, Illes Balears,
Andalucía), bsarg_s_post is zeroed so that post = bsa00_s only.

Regional exposure index:
    Exposure_r = Σ_h (w_h × Gain_h,r) / Σ_h w_h
    Pooled equally across 2017, 2018, 2019.

Excluded regions: La Rioja (23), Aragón (24) — broken RMI parameterisation
in both pre-reform and IMV counterfactual simulations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_gain(
    rmi_df: pd.DataFrame,
    imv_df: pd.DataFrame,
    year: int,
    exclude_regions: frozenset[int],
    incompatible_regions: frozenset[int],
) -> pd.DataFrame:

    rmi_cols = ["idperson", "drgn2", "dwt", "bsarg_s"]
    imv_cols = ["idperson", "bsa00_s", "bsarg_s"]

    rmi = rmi_df[rmi_cols].copy().rename(
        columns={"bsarg_s": "bsarg_s_pre"}
    )
    imv = imv_df[imv_cols].copy().rename(
        columns={"bsa00_s": "bsa00_s_post", "bsarg_s": "bsarg_s_post"}
    )

    merged = rmi.merge(imv, on="idperson", how="inner")

    n_before = len(rmi)
    n_after  = len(merged)
    if n_after < n_before:
        logger.warning(
            "Year %d: %d persons lost in merge (%d → %d)",
            year, n_before - n_after, n_before, n_after,
        )

    merged.loc[
        merged["drgn2"].isin(incompatible_regions), "bsarg_s_post"
    ] = 0.0

    # Compute post-reform total protection and gain
    merged["total_post"] = merged["bsa00_s_post"] + merged["bsarg_s_post"]
    merged["gain"]       = merged["total_post"] - merged["bsarg_s_pre"]
    merged["year"]       = year

    # Exclude broken regions
    n_before_excl = len(merged)
    merged = merged[~merged["drgn2"].isin(exclude_regions)].copy()
    n_excluded = n_before_excl - len(merged)
    if n_excluded > 0:
        logger.info(
            "Year %d: excluded %d persons from regions %s",
            year, n_excluded, sorted(exclude_regions),
        )

    logger.info(
        "Year %d: computed gain for %d persons | "
        "mean gain=€%.2f | positive=%.1f%% | negative=%.1f%%",
        year, len(merged),
        merged["gain"].mean(),
        100 * (merged["gain"] > 0).mean(),
        100 * (merged["gain"] < 0).mean(),
    )

    return merged


def pool_gains(
    rmi_dfs: dict[int, pd.DataFrame],
    imv_dfs: dict[int, pd.DataFrame],
    exclude_regions: frozenset[int],
    incompatible_regions: frozenset[int],
) -> pd.DataFrame:

    frames = []
    for year in sorted(rmi_dfs.keys()):
        gain_df = compute_gain(
            rmi_dfs[year],
            imv_dfs[year],
            year,
            exclude_regions,
            incompatible_regions,
        )
        frames.append(gain_df)

    pooled = pd.concat(frames, ignore_index=True)
    logger.info(
        "Pooled gain dataset: %d person-year observations across %d years",
        len(pooled), len(rmi_dfs),
    )
    return pooled


def compute_exposure_index(
    pooled: pd.DataFrame,
    region_names: dict[int, str],
) -> pd.DataFrame:

    def weighted_mean(x: pd.DataFrame, col: str) -> float:
        w = x["dwt"].sum()
        return (x[col] * x["dwt"]).sum() / w if w > 0 else np.nan

    regional = (
        pooled.groupby("drgn2")
        .apply(lambda x: pd.Series({
            "exposure":              round(weighted_mean(x, "gain"), 2),
            "mean_gain_pre":         round(weighted_mean(x, "bsarg_s_pre"), 2),
            "mean_total_post":       round(weighted_mean(x, "total_post"), 2),
            "mean_imv":              round(weighted_mean(x, "bsa00_s_post"), 2),
            "mean_rmi_post":         round(weighted_mean(x, "bsarg_s_post"), 2),
            "weighted_pop":          round(x["dwt"].sum() / 3, 0),  # avg per year
            "n_observations":        len(x),
        }))
        .reset_index()
        .round(2)
    )

    regional["region"] = regional["drgn2"].map(region_names)
    regional = regional.sort_values("exposure", ascending=False).reset_index(drop=True)

    logger.info("Regional exposure index computed for %d regions", len(regional))
    logger.info(
        "Exposure range: €%.2f to €%.2f",
        regional["exposure"].min(),
        regional["exposure"].max(),
    )
    logger.info(
        "Regions with positive exposure (IMV > pre-reform RMI): %d",
        (regional["exposure"] > 0).sum(),
    )
    logger.info(
        "Regions with negative exposure (IMV < pre-reform RMI): %d",
        (regional["exposure"] < 0).sum(),
    )

    return regional


def plot_exposure(
    exposure_df: pd.DataFrame,
    output_dir: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [
        "#378ADD" if v >= 0 else "#E05C5C"
        for v in exposure_df["exposure"]
    ]

    bars = ax.barh(
        exposure_df["region"],
        exposure_df["exposure"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    for bar, val in zip(bars, exposure_df["exposure"]):
        x_pos = bar.get_width() + (2 if val >= 0 else -2)
        ha    = "left" if val >= 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"€{val:.1f}",
            va="center", ha=ha, fontsize=8, color="#3A3A3A",
        )

    ax.axvline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
    ax.set_xlabel(
        "Weighted mean gain (€/month per household)\n"
        "Positive = IMV more generous than pre-reform RMI | "
        "Negative = IMV less generous",
        fontsize=9,
    )
    ax.set_title(
        "Regional exposure to the IMV reform\n"
        "Simulated gain: post-reform total protection minus pre-reform RMI\n"
        "Pooled 2017–2019, 2022 IMV rules, excl. La Rioja and Aragón",
        fontsize=10, pad=12,
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"€{int(x)}")
    )
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = output_dir / "exposure_index.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Exposure plot saved → %s", out_path)
    plt.close()


def save_exposure(
    exposure_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_cols = [
        "drgn2", "region", "exposure",
        "mean_gain_pre", "mean_total_post",
        "mean_imv", "mean_rmi_post",
        "weighted_pop", "n_observations",
    ]
    exp_path = output_dir / "exposure_index.csv"
    exposure_df[exp_cols].to_csv(exp_path, index=False)
    logger.info("Exposure index saved → %s", exp_path)

    hh_cols = [
        "idperson", "drgn2", "dwt", "year",
        "bsarg_s_pre", "bsa00_s_post", "bsarg_s_post",
        "total_post", "gain",
    ]
    hh_path = output_dir / "household_gains.csv"
    pooled_df[hh_cols].to_csv(hh_path, index=False)
    logger.info("Household gains saved → %s", hh_path)
"""
exposure_compute.py
===================
Computes the regional exposure index using PCA on three generosity dimensions.

Gain dimensions (per region r, year t):
    1. Δmean_benefit_r,t    = imv_mean_monthly_r,t - rmi_mean_monthly_r,t
    2. Δrecipients_pc_r,t   = (imv_recipients_w_r,t - rmi_recipients_w_r,t) / pop_r,t
    3. Δexpenditure_pc_r,t  = (imv_expenditure_r,t - rmi_expenditure_r,t) / pop_r,t

Where:
    imv_mean_monthly   = weighted mean of (bsa00_s + bsarg_s) among post recipients
    rmi_mean_monthly   = weighted mean of bsarg_s among pre-reform recipients
    imv_recipients_w   = weighted count of households with bsa00_s > 0
    rmi_recipients_w   = weighted count of households with bsarg_s > 0
    imv_expenditure    = Σ_h((bsa00_s + bsarg_s) × dwt_h) × 12
    rmi_expenditure    = Σ_h(bsarg_s × dwt_h) × 12
    pop_r              = Σ_h dwt_h (total weighted regional population)

Exposure index:
    - Average each dimension across 2017, 2018, 2019
    - Standardize all three dimensions (mean=0, std=1)
    - Extract first principal component via PCA
    - Exposure_r = PC1 score for region r

Excluded regions:
    La Rioja (23), Aragón (24) — broken EUROMOD J2.0+ parameterisation
    Ceuta (63) — zero pre-reform recipients, extreme outlier in delta_mean

For regions where RMI is INCOMPATIBLE with IMV:
    Galicia (11), Illes Balears (53), Andalucía (61)
    bsarg_s_post is zeroed so post_protection = bsa00_s only
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_regional_dimensions(
    rmi_df: pd.DataFrame,
    imv_df: pd.DataFrame,
    year: int,
    exclude_regions: frozenset[int],
    incompatible_regions: frozenset[int],
) -> pd.DataFrame:
    """
    Compute the three generosity dimensions for each region in one year.
    """
    imv = imv_df.copy()
    imv.loc[imv["drgn2"].isin(incompatible_regions), "bsarg_s"] = 0.0
    imv["total_post"] = imv["bsa00_s"] + imv["bsarg_s"]

    results = []

    for drgn2 in sorted(rmi_df["drgn2"].dropna().unique()):
        if drgn2 in exclude_regions:
            continue

        r   = rmi_df[rmi_df["drgn2"] == drgn2]
        i   = imv[imv["drgn2"] == drgn2]
        pop = r["dwt"].sum()

        if pop == 0:
            continue

        rmi_rec   = r[r["bsarg_s"] > 0]
        rmi_rec_w = rmi_rec["dwt"].sum()
        rmi_mean  = (
            (rmi_rec["bsarg_s"] * rmi_rec["dwt"]).sum() / rmi_rec_w
            if rmi_rec_w > 0 else 0.0
        )
        rmi_exp   = (r["bsarg_s"] * r["dwt"]).sum() * 12

        imv_rec    = i[i["bsa00_s"] > 0]
        imv_rec_w  = imv_rec["dwt"].sum()
        post_pos   = i[i["total_post"] > 0]
        imv_mean   = (
            (post_pos["total_post"] * post_pos["dwt"]).sum() /
            post_pos["dwt"].sum()
            if post_pos["dwt"].sum() > 0 else 0.0
        )
        imv_exp    = (i["total_post"] * i["dwt"]).sum() * 12

        results.append({
            "drgn2":               int(drgn2),
            "year":                year,
            "pop":                 pop,
            "rmi_mean":            round(rmi_mean, 2),
            "imv_mean":            round(imv_mean, 2),
            "delta_mean":          round(imv_mean - rmi_mean, 2),
            "rmi_recipients_w":    round(rmi_rec_w, 0),
            "imv_recipients_w":    round(imv_rec_w, 0),
            "delta_recipients_pc": round(
                (imv_rec_w - rmi_rec_w) / pop * 100, 4
            ),
            "rmi_expenditure":     round(rmi_exp, 0),
            "imv_expenditure":     round(imv_exp, 0),
            "delta_expenditure_pc": round(
                (imv_exp - rmi_exp) / pop, 4
            ),
        })

    df = pd.DataFrame(results)
    logger.info(
        "Year %d: computed dimensions for %d regions", year, len(df)
    )
    return df


def pool_dimensions(
    rmi_dfs: dict[int, pd.DataFrame],
    imv_dfs: dict[int, pd.DataFrame],
    exclude_regions: frozenset[int],
    incompatible_regions: frozenset[int],
) -> pd.DataFrame:
    """
    Compute dimensions for each year and average across 2017–2019.
    """
    frames = []
    for year in sorted(rmi_dfs.keys()):
        frames.append(
            compute_regional_dimensions(
                rmi_dfs[year], imv_dfs[year], year,
                exclude_regions, incompatible_regions,
            )
        )

    all_dims = pd.concat(frames, ignore_index=True)

    avg = (
        all_dims.groupby("drgn2")
        .agg(
            pop                   = ("pop",                  "mean"),
            rmi_mean              = ("rmi_mean",              "mean"),
            imv_mean              = ("imv_mean",              "mean"),
            delta_mean            = ("delta_mean",            "mean"),
            rmi_recipients_w      = ("rmi_recipients_w",      "mean"),
            imv_recipients_w      = ("imv_recipients_w",      "mean"),
            delta_recipients_pc   = ("delta_recipients_pc",   "mean"),
            rmi_expenditure       = ("rmi_expenditure",       "mean"),
            imv_expenditure       = ("imv_expenditure",       "mean"),
            delta_expenditure_pc  = ("delta_expenditure_pc",  "mean"),
        )
        .reset_index()
        .round(4)
    )

    logger.info(
        "Pooled %d years → %d regions", len(rmi_dfs), len(avg)
    )
    return avg


def compute_pca_exposure(
    pooled: pd.DataFrame,
    region_names: dict[int, str],
) -> pd.DataFrame:
    """
    Standardize the three gain dimensions and extract PC1 as exposure index.

    PCA inputs (all averaged across 2017–2019):
        delta_mean           — Δ mean monthly benefit (€/month)
        delta_recipients_pc  — Δ recipients per capita (%)
        delta_expenditure_pc — Δ annual expenditure per capita (€)

    Returns
    -------
    pd.DataFrame sorted by exposure (PC1) descending.
    PCA diagnostics stored in df.attrs.
    """
    dims     = ["delta_mean", "delta_recipients_pc", "delta_expenditure_pc"]
    X        = pooled[dims].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca    = PCA(n_components=3)
    scores = pca.fit_transform(X_scaled)

    result = pooled.copy()
    result["exposure"] = scores[:, 0]
    result["pc2"]      = scores[:, 1]
    result["pc3"]      = scores[:, 2]
    result["region"]   = result["drgn2"].map(region_names)

    explained = pca.explained_variance_ratio_
    loadings  = pca.components_

    logger.info(
        "PCA explained variance: PC1=%.1f%% PC2=%.1f%% PC3=%.1f%%",
        100*explained[0], 100*explained[1], 100*explained[2],
    )
    logger.info("PC1 loadings:")
    for dim, loading in zip(dims, loadings[0]):
        logger.info("  %-28s : %+.3f", dim, loading)

    if explained[0] < 0.50:
        logger.warning(
            "PC1 explains only %.1f%% of variance — "
            "three dimensions have low common variation. "
            "Consider reporting all dimensions separately.",
            100 * explained[0],
        )

    result.attrs["pca_explained"] = explained
    result.attrs["pca_loadings"]  = dict(zip(dims, loadings[0]))
    result.attrs["pca_means"]     = dict(zip(dims, scaler.mean_))
    result.attrs["pca_stds"]      = dict(zip(dims, scaler.scale_))

    return result.sort_values("exposure", ascending=False).reset_index(drop=True)


def plot_exposure(
    exposure_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Two-panel figure:
      Left  — horizontal bar chart of PC1 scores by region
      Right — PC1 loadings bar chart
    """
    explained = exposure_df.attrs.get("pca_explained", [0, 0, 0])
    loadings  = exposure_df.attrs.get("pca_loadings", {})

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    ax = axes[0]
    colors = [
        "#378ADD" if v >= 0 else "#E05C5C"
        for v in exposure_df["exposure"]
    ]
    bars = ax.barh(
        exposure_df["region"],
        exposure_df["exposure"],
        color=colors, edgecolor="white", linewidth=0.5, height=0.7,
    )
    for bar, val in zip(bars, exposure_df["exposure"]):
        x_pos = bar.get_width() + (0.02 if val >= 0 else -0.02)
        ha    = "left" if val >= 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha=ha,
            fontsize=8, color="#3A3A3A",
        )
    ax.axvline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
    ax.set_xlabel(
        f"PC1 score (explains {100*explained[0]:.1f}% of variance)\n"
        "Positive = IMV more generous than pre-reform RMI | "
        "Negative = IMV less generous",
        fontsize=9,
    )
    ax.set_title(
        "Regional exposure to IMV reform — PCA composite index\n"
        "Pooled 2017–2019, 2022 IMV rules\n"
        "excl. La Rioja, Aragón, Ceuta",
        fontsize=10, pad=10,
    )
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.invert_yaxis()

    ax2 = axes[1]
    dim_labels = {
        "delta_mean":           "Δ Mean benefit\n(€/month)",
        "delta_recipients_pc":  "Δ Recipients\n(% pop)",
        "delta_expenditure_pc": "Δ Expenditure\n(€/capita)",
    }
    load_vals   = list(loadings.values())
    load_labels = [dim_labels.get(k, k) for k in loadings]
    colors2     = ["#378ADD" if v >= 0 else "#E05C5C" for v in load_vals]
    ax2.barh(
        load_labels, load_vals,
        color=colors2, edgecolor="white", height=0.5,
    )
    ax2.axvline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("PC1 loading", fontsize=9)
    ax2.set_title(
        "PC1 loadings\n(contribution of\neach dimension)",
        fontsize=9, pad=10,
    )
    ax2.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax2.set_xlim(-1.1, 1.1)

    plt.tight_layout()
    out_path = output_dir / "exposure_index_pca.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("PCA exposure plot saved → %s", out_path)
    plt.close()


def save_exposure(
    exposure_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save exposure index and PCA diagnostics to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    out_cols = [
        "drgn2", "region",
        "exposure",
        "delta_mean", "delta_recipients_pc", "delta_expenditure_pc",
        "rmi_mean", "imv_mean",
        "rmi_recipients_w", "imv_recipients_w",
        "rmi_expenditure", "imv_expenditure",
        "pop",
    ]
    exp_path = output_dir / "exposure_index.csv"
    exposure_df[out_cols].to_csv(exp_path, index=False)
    logger.info("Exposure index saved → %s", exp_path)

    explained = exposure_df.attrs.get("pca_explained", [])
    loadings  = exposure_df.attrs.get("pca_loadings", {})
    means     = exposure_df.attrs.get("pca_means", {})
    stds      = exposure_df.attrs.get("pca_stds", {})

    pd.DataFrame({
        "component": ["PC1", "PC2", "PC3"],
        "explained_variance_pct": [round(100*e, 2) for e in explained],
    }).to_csv(output_dir / "pca_diagnostics.csv", index=False)

    pd.DataFrame([
        {
            "dimension": k,
            "pc1_loading": round(v, 4),
            "raw_mean":    round(means.get(k, np.nan), 4),
            "raw_std":     round(stds.get(k, np.nan),  4),
        }
        for k, v in loadings.items()
    ]).to_csv(output_dir / "pca_loadings.csv", index=False)

    logger.info(
        "PCA diagnostics saved → %s",
        output_dir / "pca_diagnostics.csv",
    )
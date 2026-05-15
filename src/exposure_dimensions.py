"""
exposure_dimensions.py
======================
Computes and pools raw regional gain dimensions for the IMV exposure pipeline.

Responsibilities
----------------
1. compute_regional_dimensions() — per-year, per-region raw simulated values
2. pool_dimensions()             — average before differencing, merge
                                   administrative Informe data, compute
                                   all delta dimensions

Design principle — average before differencing
-----------------------------------------------
Raw annual values are averaged across 2017, 2018, 2019 BEFORE computing
deltas. Differencing noisy yearly estimates first and then averaging
compounds sampling error; averaging the raw series first reduces noise
in each component by √3 before the subtraction.

Delta dimensions produced
--------------------------
Hybrid (IMV simulated vs RMI administrative):
    delta_exp_hybrid   = (avg_imv_exp_sim - avg_rmi_exp_admin) / avg_pop
    delta_cov_hybrid   = (avg_imv_rec_sim - avg_titulares_admin) / avg_pop × 100

Fully simulated (both sides from EUROMOD):
    delta_exp_sim      = (avg_imv_exp_sim - avg_rmi_exp_sim) / avg_pop
    delta_cov_sim      = (avg_imv_rec_sim - avg_rmi_rec_sim) / avg_pop × 100

Purely administrative (no simulation):
    delta_exp_admin    = (-avg_rmi_exp_admin) / avg_pop
    delta_cov_admin    = (-avg_titulares_admin) / avg_pop × 100

Descriptive only (not used in any exposure spec):
    delta_mean         = avg_imv_mean - avg_rmi_mean

Per-year delta for Test 4 stability (stored in all_dims only):
    delta_exp_sim_yr   = (imv_exp_sim_t - rmi_exp_sim_t) / pop_t

For incompatible regions (Galicia 11, Illes Balears 53, Andalucía 61):
    bsarg_s in the IMV run is zeroed → post_protection = bsa00_s only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_regional_dimensions(
    rmi_df: pd.DataFrame,
    imv_df: pd.DataFrame,
    year: int,
    exclude_regions: frozenset[int],
    incompatible_regions: frozenset[int],
) -> pd.DataFrame:
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

        imv_rec   = i[i["bsa00_s"] > 0]
        imv_rec_w = imv_rec["dwt"].sum()
        post_pos  = i[i["total_post"] > 0]
        imv_mean  = (
            (post_pos["total_post"] * post_pos["dwt"]).sum() /
            post_pos["dwt"].sum()
            if post_pos["dwt"].sum() > 0 else 0.0
        )
        imv_exp   = (i["total_post"] * i["dwt"]).sum() * 12

        results.append({
            "drgn2":             int(drgn2),
            "year":              year,
            "pop":               pop,
            "rmi_exp_sim":       round(rmi_exp,   0),
            "imv_exp_sim":       round(imv_exp,   0),
            "rmi_rec_sim":       round(rmi_rec_w, 0),
            "imv_rec_sim":       round(imv_rec_w, 0),
            "rmi_mean_sim":      round(rmi_mean,  2),
            "imv_mean_sim":      round(imv_mean,  2),
            "delta_exp_sim_yr":  round(
                (imv_exp - rmi_exp) / pop, 4
            ) if pop > 0 else np.nan,
        })

    df = pd.DataFrame(results)
    logger.info(
        "Year %d: computed raw dimensions for %d regions", year, len(df)
    )
    return df


def pool_dimensions(
    rmi_dfs: dict[int, pd.DataFrame],
    imv_dfs: dict[int, pd.DataFrame],
    exclude_regions: frozenset[int],
    incompatible_regions: frozenset[int],
    informe_rmi: dict[int, list[dict]],
    region_population: dict[int, dict[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for year in sorted(rmi_dfs.keys()):
        frames.append(
            compute_regional_dimensions(
                rmi_dfs[year], imv_dfs[year], year,
                exclude_regions, incompatible_regions,
            )
        )
    all_dims = pd.concat(frames, ignore_index=True)

    # --- Step 1: average raw simulated values across years ---
    pooled = (
        all_dims.groupby("drgn2")
        .agg(
            pop         = ("pop",         "mean"),
            rmi_exp_sim = ("rmi_exp_sim", "mean"),
            imv_exp_sim = ("imv_exp_sim", "mean"),
            rmi_rec_sim = ("rmi_rec_sim", "mean"),
            imv_rec_sim = ("imv_rec_sim", "mean"),
            rmi_mean_sim= ("rmi_mean_sim","mean"),
            imv_mean_sim= ("imv_mean_sim","mean"),
        )
        .reset_index()
        .round(4)
    )

    admin_records = []
    years = sorted(informe_rmi.keys())
    for year in years:
        pop_year = region_population.get(year, {})
        for row in informe_rmi[year]:
            drgn2 = row["drgn2"]
            if drgn2 in exclude_regions:
                continue
            admin_records.append({
                "drgn2":         drgn2,
                "year":          year,
                "rmi_exp_admin": row["gasto_anual_ejecutado"],
                "titulares":     row["titulares"],
                "pop_admin":     pop_year.get(drgn2, np.nan),
            })

    admin_df = (
        pd.DataFrame(admin_records)
        .groupby("drgn2")
        .agg(
            avg_rmi_exp_admin  = ("rmi_exp_admin", "mean"),
            avg_titulares_admin= ("titulares",      "mean"),
            avg_pop_admin      = ("pop_admin",      "mean"),
        )
        .reset_index()
        .round(2)
    )

    pooled = pooled.merge(admin_df, on="drgn2", how="left")

    pooled["delta_exp_hybrid"] = (
        (pooled["imv_exp_sim"] - pooled["avg_rmi_exp_admin"]) /
        pooled["avg_pop_admin"]
    ).round(4)

    pooled["delta_cov_hybrid"] = (
        (pooled["imv_rec_sim"] - pooled["avg_titulares_admin"]) /
        pooled["avg_pop_admin"] * 100
    ).round(4)

    pooled["delta_exp_sim"] = (
        (pooled["imv_exp_sim"] - pooled["rmi_exp_sim"]) /
        pooled["pop"]
    ).round(4)

    pooled["delta_cov_sim"] = (
        (pooled["imv_rec_sim"] - pooled["rmi_rec_sim"]) /
        pooled["pop"] * 100
    ).round(4)

    pooled["delta_exp_admin"] = (
        pooled["avg_rmi_exp_admin"] / pooled["pop"]
    ).round(4)

    pooled["delta_cov_admin"] = (
        - pooled["avg_titulares_admin"] / pooled["pop"] * 100
    ).round(4)

    pooled["delta_mean"] = (
        - pooled["imv_mean_sim"] - pooled["rmi_mean_sim"]
    ).round(2)

    logger.info(
        "Pooled %d years → %d regions (average before differencing)",
        len(rmi_dfs), len(pooled),
    )
    logger.info(
        "\nPooled dimensions:\n%s",
        pooled[[
            "drgn2",
            "delta_exp_hybrid", "delta_cov_hybrid",
            "delta_exp_sim",    "delta_cov_sim",
            "delta_exp_admin",  "delta_cov_admin",
            "delta_mean",
        ]].to_string(index=False),
    )
    return pooled, all_dims
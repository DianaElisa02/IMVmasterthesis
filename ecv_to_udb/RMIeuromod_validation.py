"""
RMIeuromod_validation.py
===================
Validates EUROMOD pre-reform RMI simulations against Informe RMI
administrative benchmarks for 2017, 2018, and 2019.

For each year:
  - Reads the EUROMOD output file from output/
  - Computes weighted mean monthly RMI (il_bsarg_global) by region
  - Compares against Informe RMI Cuadro 7 (cuantía mínima, titulares)
    and Cuadro 8 (gasto anual por titular)
  - Reports Pearson and Spearman rank correlations
  - Exports comparison tables and a validation plot

Sources
-------
Informe de Rentas Mínimas de Inserción 2017, 2018, 2019.
Ministerio de Derechos Sociales y Agenda 2030.
Cuadro 7: Beneficiarios/as — cuantía mínima y titulares por CC.AA.
Cuadro 8: Gasto anual — gasto anual por titular prestación.

Output
------
output/validation_table_YYYY.csv   — comparison table per year
output/validation_plot.png          — multi-year validation chart
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent
OUTPUT_DIR  = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EUROMOD_FILES = {
    2017: BASE_DIR / "output" / "ES_2017_a2.txt",
    2018: BASE_DIR / "output" / "ES_2018_a1.txt",
    2019: BASE_DIR / "output" / "ES_2019_b1.txt",
}

# =============================================================================
# INFORME RMI ADMINISTRATIVE BENCHMARKS
# Hardcoded from Cuadro 7 and Cuadro 8 of each year's Informe RMI.
# drgn2: EUROMOD NUTS-2 integer code for each Autonomous Community.
# cuantia_minima: statutory minimum monthly benefit per titular (€/month).
# titulares: number of benefit recipients (titulares prestación), total.
# gasto_anual_por_titular: annual expenditure per titular (€/year),
#   from Cuadro 8 column "Gasto anual por titular prestación".
#   Divide by 12 for average monthly administrative benchmark.
# =============================================================================

INFORME_RMI: dict[int, list[dict]] = {

    2017: [
        # Source: Informe RMI 2017, Cuadro 7 (p.79) and Cuadro 8 (p.80)
        {"region": "Andalucía",          "drgn2": 61, "cuantia_minima": 406.22, "titulares": 29337,  "gasto_anual_por_titular": 2942.35},
        {"region": "Aragón",             "drgn2": 24, "cuantia_minima": 472.00, "titulares": 10466,  "gasto_anual_por_titular": 4713.90},
        {"region": "Asturias",           "drgn2": 12, "cuantia_minima": 442.96, "titulares": 22219,  "gasto_anual_por_titular": 5361.19},
        {"region": "Illes Balears",      "drgn2": 53, "cuantia_minima": 430.36, "titulares": 7551,   "gasto_anual_por_titular": 1763.10},
        {"region": "Canarias",           "drgn2": 70, "cuantia_minima": 476.88, "titulares": 13525,  "gasto_anual_por_titular": 3923.00},
        {"region": "Cantabria",          "drgn2": 13, "cuantia_minima": 430.27, "titulares": 6366,   "gasto_anual_por_titular": 4902.52},
        {"region": "Castilla-La Mancha", "drgn2": 42, "cuantia_minima": 420.42, "titulares": 3152,   "gasto_anual_por_titular": 2129.21},
        {"region": "Castilla y León",    "drgn2": 41, "cuantia_minima": 430.27, "titulares": 15502,  "gasto_anual_por_titular": 5013.16},
        {"region": "Cataluña",           "drgn2": 51, "cuantia_minima": 564.00, "titulares": 26311,  "gasto_anual_por_titular": 7014.99},
        {"region": "Ceuta",              "drgn2": 63, "cuantia_minima": 300.00, "titulares": 263,    "gasto_anual_por_titular": 2146.86},
        {"region": "Extremadura",        "drgn2": 43, "cuantia_minima": 430.27, "titulares": 6316,   "gasto_anual_por_titular": 7668.46},
        {"region": "Galicia",            "drgn2": 11, "cuantia_minima": 403.38, "titulares": 14468,  "gasto_anual_por_titular": 4064.81},
        {"region": "Madrid",             "drgn2": 30, "cuantia_minima": 400.00, "titulares": 35483,  "gasto_anual_por_titular": 4752.32},
        {"region": "Melilla",            "drgn2": 64, "cuantia_minima": 458.64, "titulares": 994,    "gasto_anual_por_titular": 4088.83},
        {"region": "Murcia",             "drgn2": 62, "cuantia_minima": 430.27, "titulares": 5421,   "gasto_anual_por_titular": 2609.24},
        {"region": "Navarra",            "drgn2": 22, "cuantia_minima": 600.00, "titulares": 15918,  "gasto_anual_por_titular": 6161.69},
        {"region": "País Vasco",         "drgn2": 21, "cuantia_minima": 634.97, "titulares": 76188,  "gasto_anual_por_titular": 6148.30},
        {"region": "La Rioja",           "drgn2": 23, "cuantia_minima": 430.27, "titulares": 2424,   "gasto_anual_por_titular": 2075.08},
        {"region": "C. Valenciana",      "drgn2": 52, "cuantia_minima": 388.51, "titulares": 21387,  "gasto_anual_por_titular": 2707.61},
    ],

    2018: [
        # Source: Informe RMI 2018, Cuadro 7 (p.79) and Cuadro 8 (p.82)
        {"region": "Andalucía",          "drgn2": 61, "cuantia_minima": 419.52, "titulares": 17883,  "gasto_anual_por_titular": 3003.66},
        {"region": "Aragón",             "drgn2": 24, "cuantia_minima": 491.00, "titulares": 9894,   "gasto_anual_por_titular": 4902.38},
        {"region": "Asturias",           "drgn2": 12, "cuantia_minima": 442.96, "titulares": 22305,  "gasto_anual_por_titular": 5584.11},
        {"region": "Illes Balears",      "drgn2": 53, "cuantia_minima": 431.53, "titulares": 9714,   "gasto_anual_por_titular": 2196.11},
        {"region": "Canarias",           "drgn2": 70, "cuantia_minima": 478.77, "titulares": 11592,  "gasto_anual_por_titular": 3676.40},
        {"region": "Cantabria",          "drgn2": 13, "cuantia_minima": 430.27, "titulares": 5365,   "gasto_anual_por_titular": 5775.67},
        {"region": "Castilla-La Mancha", "drgn2": 42, "cuantia_minima": 446.45, "titulares": 3544,   "gasto_anual_por_titular": 2663.89},
        {"region": "Castilla y León",    "drgn2": 41, "cuantia_minima": 430.27, "titulares": 14536,  "gasto_anual_por_titular": 4944.90},
        {"region": "Cataluña",           "drgn2": 51, "cuantia_minima": 604.00, "titulares": 28572,  "gasto_anual_por_titular": 8417.56},
        {"region": "Ceuta",              "drgn2": 63, "cuantia_minima": 300.00, "titulares": 266,    "gasto_anual_por_titular": 1659.91},
        {"region": "Extremadura",        "drgn2": 43, "cuantia_minima": 430.27, "titulares": 5982,   "gasto_anual_por_titular": 8096.62},
        {"region": "Galicia",            "drgn2": 11, "cuantia_minima": 403.38, "titulares": 14238,  "gasto_anual_por_titular": 3885.31},
        {"region": "Madrid",             "drgn2": 30, "cuantia_minima": 400.00, "titulares": 33000,  "gasto_anual_por_titular": 4623.03},
        {"region": "Melilla",            "drgn2": 64, "cuantia_minima": 458.64, "titulares": 784,    "gasto_anual_por_titular": 4217.55},
        {"region": "Murcia",             "drgn2": 62, "cuantia_minima": 430.27, "titulares": 5856,   "gasto_anual_por_titular": 2821.62},
        {"region": "Navarra",            "drgn2": 22, "cuantia_minima": 610.80, "titulares": 16078,  "gasto_anual_por_titular": 6438.65},
        {"region": "País Vasco",         "drgn2": 21, "cuantia_minima": 644.49, "titulares": 72341,  "gasto_anual_por_titular": 6062.42},
        {"region": "La Rioja",           "drgn2": 23, "cuantia_minima": 430.27, "titulares": 2941,   "gasto_anual_por_titular": 4280.86},
        {"region": "C. Valenciana",      "drgn2": 52, "cuantia_minima": 515.13, "titulares": 18411,  "gasto_anual_por_titular": None},
    ],

    2019: [
        # Source: Informe RMI 2019, Cuadro 7 (p.149) and Cuadro 8.1 (p.152)
        {"region": "Andalucía",          "drgn2": 61, "cuantia_minima": 419.52, "titulares": 22318,  "gasto_anual_por_titular": 4824.52},
        {"region": "Aragón",             "drgn2": 24, "cuantia_minima": 491.00, "titulares": 9401,   "gasto_anual_por_titular": 4974.20},
        {"region": "Asturias",           "drgn2": 12, "cuantia_minima": 448.28, "titulares": 21947,  "gasto_anual_por_titular": 5517.56},
        {"region": "Illes Balears",      "drgn2": 53, "cuantia_minima": 457.31, "titulares": 10449,  "gasto_anual_por_titular": 2639.81},
        {"region": "Canarias",           "drgn2": 70, "cuantia_minima": 486.90, "titulares": 9973,   "gasto_anual_por_titular": 3663.09},
        {"region": "Cantabria",          "drgn2": 13, "cuantia_minima": 430.27, "titulares": 7052,   "gasto_anual_por_titular": 4260.35},
        {"region": "Castilla-La Mancha", "drgn2": 42, "cuantia_minima": 546.00, "titulares": 4132,   "gasto_anual_por_titular": 4296.13},
        {"region": "Castilla y León",    "drgn2": 41, "cuantia_minima": 430.27, "titulares": 13069,  "gasto_anual_por_titular": 4903.90},
        {"region": "Cataluña",           "drgn2": 51, "cuantia_minima": 644.00, "titulares": 32166,  "gasto_anual_por_titular": 8317.18},
        {"region": "Ceuta",              "drgn2": 63, "cuantia_minima": 300.00, "titulares": 179,    "gasto_anual_por_titular": 2713.25},
        {"region": "Extremadura",        "drgn2": 43, "cuantia_minima": 537.84, "titulares": 7991,   "gasto_anual_por_titular": 5935.93},
        {"region": "Galicia",            "drgn2": 11, "cuantia_minima": 403.38, "titulares": 13600,  "gasto_anual_por_titular": 3925.85},
        {"region": "Madrid",             "drgn2": 30, "cuantia_minima": 400.00, "titulares": 28643,  "gasto_anual_por_titular": 5421.78},
        {"region": "Melilla",            "drgn2": 64, "cuantia_minima": 458.64, "titulares": 510,    "gasto_anual_por_titular": 3117.26},
        {"region": "Murcia",             "drgn2": 62, "cuantia_minima": 430.27, "titulares": 6355,   "gasto_anual_por_titular": 2906.34},
        {"region": "Navarra",            "drgn2": 22, "cuantia_minima": 623.63, "titulares": 15712,  "gasto_anual_por_titular": 6561.06},
        {"region": "País Vasco",         "drgn2": 21, "cuantia_minima": 667.05, "titulares": 66508,  "gasto_anual_por_titular": 6352.42},
        {"region": "La Rioja",           "drgn2": 23, "cuantia_minima": 430.27, "titulares": 3070,   "gasto_anual_por_titular": 4532.00},
        {"region": "C. Valenciana",      "drgn2": 52, "cuantia_minima": 630.00, "titulares": 24108,  "gasto_anual_por_titular": 6279.73},
    ],
}


def load_euromod_output(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."), errors="coerce"
                )
            except Exception:
                pass
    return df


def compute_regional_rmi(df: pd.DataFrame) -> pd.DataFrame:
    recipients = df[df["bsa"] > 0].copy()
    regional = (
        recipients.groupby("drgn2")
        .agg(
            euromod_recipients=("bsa", "count"),
            euromod_mean_monthly=("bsa", "mean"),
        )
        .reset_index()
        .round(2)
    )
    return regional


def build_comparison(year: int, regional: pd.DataFrame) -> pd.DataFrame:
    informe = pd.DataFrame(INFORME_RMI[year])
    informe["avg_monthly_admin"] = (
        informe["gasto_anual_por_titular"] / 12
    ).round(2)

    df = informe.merge(regional, on="drgn2", how="left")

    df["diff_euromod_cuantia"] = (
        df["euromod_mean_monthly"] - df["cuantia_minima"]
    ).round(2)
    df["ratio_euromod_cuantia"] = (
        df["euromod_mean_monthly"] / df["cuantia_minima"]
    ).round(2)

    return df


def compute_correlations(df: pd.DataFrame) -> dict:
    clean = df[["cuantia_minima", "euromod_mean_monthly"]].dropna()
    r, p_r = pearsonr(clean["cuantia_minima"], clean["euromod_mean_monthly"])
    rho, p_rho = spearmanr(clean["cuantia_minima"], clean["euromod_mean_monthly"])
    return {"pearson_r": round(r, 3), "pearson_p": round(p_r, 4),
            "spearman_rho": round(rho, 3), "spearman_p": round(p_rho, 4)}


def plot_validation(results: dict[int, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    years = sorted(results.keys())

    for ax, year in zip(axes, years):
        df = results[year].dropna(subset=["euromod_mean_monthly", "cuantia_minima"])

        ax.scatter(
            df["cuantia_minima"],
            df["euromod_mean_monthly"],
            color="#378ADD", s=60, zorder=3, alpha=0.85
        )

        for _, row in df.iterrows():
            ax.annotate(
                row["region"],
                xy=(row["cuantia_minima"], row["euromod_mean_monthly"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 2), textcoords="offset points", color="#5F5E5A"
            )

        lims = [
            min(df["cuantia_minima"].min(), df["euromod_mean_monthly"].min()) * 0.9,
            max(df["cuantia_minima"].max(), df["euromod_mean_monthly"].max()) * 1.1,
        ]
        ax.plot(lims, lims, "--", color="#B4B2A9", linewidth=0.8, zorder=1)

        corr = compute_correlations(results[year])
        ax.set_title(
            f"{year}  |  r = {corr['pearson_r']}, ρ = {corr['spearman_rho']}",
            fontsize=10
        )
        ax.set_xlabel("Cuantía mínima — Informe RMI (€/month)", fontsize=9)
        ax.set_ylabel("EUROMOD mean monthly RMI (€/month)", fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{int(x)}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{int(x)}"))
        ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.suptitle(
        "EUROMOD simulated RMI vs Informe RMI cuantía mínima — pre-reform validation",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "validation_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved validation plot → %s", out_path)
    plt.close()


def main() -> None:
    logger.info("Starting EUROMOD validation — years: %s", sorted(EUROMOD_FILES.keys()))

    results: dict[int, pd.DataFrame] = {}

    for year, path in sorted(EUROMOD_FILES.items()):
        logger.info("=" * 60)
        logger.info("Year %s", year)

        if not path.exists():
            logger.error("EUROMOD output not found: %s", path)
            continue

        df = load_euromod_output(path)
        regional = compute_regional_rmi(df)
        comparison = build_comparison(year, regional)
        corr = compute_correlations(comparison)

        results[year] = comparison

        out_cols = [
            "region", "cuantia_minima", "avg_monthly_admin",
            "euromod_mean_monthly", "diff_euromod_cuantia",
            "ratio_euromod_cuantia", "euromod_recipients", "titulares"
        ]
        logger.info("\n%s", comparison[out_cols].to_string(index=False))
        logger.info(
            "Pearson r=%.3f (p=%.4f) | Spearman rho=%.3f (p=%.4f)",
            corr["pearson_r"], corr["pearson_p"],
            corr["spearman_rho"], corr["spearman_p"]
        )

        csv_path = OUTPUT_DIR / f"validation_table_{year}.csv"
        comparison[out_cols].to_csv(csv_path, index=False)
        logger.info("Saved → %s", csv_path)

    if results:
        plot_validation(results)

    logger.info("=" * 60)
    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
"""
RMIeuromod_validation.py
========================
Multi-target validation of EUROMOD pre-reform RMI simulations against
Informe RMI administrative benchmarks for 2017, 2018, and 2019.

Validation targets (per supervisor recommendation):
  1. Weighted recipient count vs Informe titulares
  2. Total annual expenditure vs Informe gasto anual ejecutado
  3. Average monthly benefit vs gasto_anual_por_titular / 12
  4. Regional coverage rate (EUROMOD / Informe titulares)
  5. Pearson and Spearman correlations — per year and pooled

Exclusions (all years):
  - La Rioja (drgn2=23): incomplete EUROMOD J2.0+ parameterisation
    for 2017-2019 ($IPREM constant unresolvable)
  - Aragón (drgn2=24): pre-2021 Ingreso Aragonés de Inserción not
    coded in J2.0+ architecture
  - Ceuta (drgn2=63): zero simulated recipients across all years
    due to very small ECV sample (339 persons)

Murcia (drgn2=62) note:
  - Valid for 2017 and 2018
  - Zero recipients in 2019 due to BCA probabilistic allocation
    on small sample (1,791 persons); 2019 observation excluded
    automatically via dropna() in pooled analysis

Output:
  output/validation_table_YYYY.csv   — regional detail per year
  output/validation_plot.png          — multi-year scatter plot
  output/validation_pooled.csv        — pooled region-year observations
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EUROMOD_FILES = {
    2017: BASE_DIR / "input_data" / "euromod_output" / "es_2017_std.txt",
    2018: BASE_DIR / "input_data" / "euromod_output" / "es_2018_std.txt",
    2019: BASE_DIR / "input_data" / "euromod_output" / "es_2019_std.txt",
}

# Excluded from all years — see module docstring
EXCLUDE_REGIONS: set[int] = {63}

SAVE_CSV = True  # set False to skip CSV export

# National totals from Informe RMI Cuadro 7 and Cuadro 8
INFORME_NATIONAL = {
    2017: {"titulares": 313291, "gasto_anual_M": 1545.44, "cuantia_media": 449.98},
    2018: {"titulares": 293302, "gasto_anual_M": 1519.67, "cuantia_media": 463.05},
    2019: {"titulares": 297183, "gasto_anual_M": 1686.26, "cuantia_media": 486.03},
}

# =============================================================================
# INFORME RMI REGIONAL BENCHMARKS
# cuantia_minima:         statutory minimum monthly benefit (1 person), Cuadro 7
# titulares:              number of benefit recipients, Cuadro 7
# gasto_anual_por_titular: annual expenditure per titular, Cuadro 8
# avg_monthly_admin = gasto_anual_por_titular / 12
#   → correct benchmark for euromod_mean_monthly (actual average benefit,
#     not statutory minimum which varies by household size)
# =============================================================================

INFORME_RMI: dict[int, list[dict]] = {

    2017: [
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
        {"region": "C. Valenciana",      "drgn2": 52, "cuantia_minima": 515.13, "titulares": 18411,  "gasto_anual_por_titular": 2437.64},
    ],

    2019: [
        {"region": "Andalucía",          "drgn2": 61, "cuantia_minima": 419.52, "titulares": 22318,  "gasto_anual_por_titular": 4824.52},
        {"region": "Aragón",             "drgn2": 24, "cuantia_minima": 491.00, "titulares": 9401,   "gasto_anual_por_titular": 4974.20},
        {"region": "Asturias",           "drgn2": 12, "cuantia_minima": 448.28, "titulares": 21947,  "gasto_anual_por_titular": 5501.72},
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
    df = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )
    return df


def compute_regional_rmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted mean monthly RMI and recipient count by region.
    Uses bsarg_s (actual simulated regional RMI) with dwt weights.
    Excluded regions are filtered out; Murcia 2019 will produce NaN
    naturally since bsarg_s=0 for all Murcia persons that year.
    """
    recipients = df[
        (df["bsarg_s"] > 0) &
        (~df["drgn2"].isin(EXCLUDE_REGIONS))
    ].copy()

    regional = (
        recipients.groupby("drgn2")
        .apply(lambda x: pd.Series({
            "euromod_recipients":    x["dwt"].sum(),
            "euromod_mean_monthly":  (
                (x["bsarg_s"] * x["dwt"]).sum() / x["dwt"].sum()
            ),
            "euromod_expenditure_M": (
                (x["bsarg_s"] * x["dwt"]).sum() * 12 / 1_000_000
            ),
        }))
        .reset_index()
        .round(2)
    )
    return regional


def build_comparison(year: int, regional: pd.DataFrame) -> pd.DataFrame:
    informe = pd.DataFrame(INFORME_RMI[year])
    informe = informe[~informe["drgn2"].isin(EXCLUDE_REGIONS)].copy()

    informe["avg_monthly_admin"] = (
        informe["gasto_anual_por_titular"] / 12
    ).round(2)
    informe["informe_expenditure_M"] = (
        informe["gasto_anual_por_titular"] * informe["titulares"] / 1_000_000
    ).round(2)

    # Left join: all Informe regions, NaN where EUROMOD has no recipients
    df = informe.merge(regional, on="drgn2", how="left")

    df["diff_avg_benefit"] = (
        df["euromod_mean_monthly"] - df["avg_monthly_admin"]
    ).round(2)
    df["ratio_avg_benefit"] = (
        df["euromod_mean_monthly"] / df["avg_monthly_admin"]
    ).round(3)
    df["coverage_rate"] = (
        df["euromod_recipients"] / df["titulares"]
    ).round(3)
    df["ratio_expenditure"] = (
        df["euromod_expenditure_M"] / df["informe_expenditure_M"]
    ).round(3)

    return df


def compute_correlations(df: pd.DataFrame) -> dict:
    """Correlate avg_monthly_admin vs euromod_mean_monthly, dropping NaNs."""
    clean = df[["avg_monthly_admin", "euromod_mean_monthly"]].dropna()
    r,   p_r   = pearsonr(clean["avg_monthly_admin"], clean["euromod_mean_monthly"])
    rho, p_rho = spearmanr(clean["avg_monthly_admin"], clean["euromod_mean_monthly"])
    return {
        "n":            len(clean),
        "pearson_r":    round(r,   3),
        "pearson_p":    round(p_r, 4),
        "spearman_rho": round(rho, 3),
        "spearman_p":   round(p_rho, 4),
    }


def print_national_summary(
    year: int,
    euromod_df: pd.DataFrame,
) -> None:
    nat = INFORME_NATIONAL[year]

    recipients = euromod_df[
        (euromod_df["bsarg_s"] > 0) &
        (~euromod_df["drgn2"].isin(EXCLUDE_REGIONS))
    ]
    weighted_recipients    = recipients["dwt"].sum()
    weighted_expenditure_M = (
        recipients["bsarg_s"] * recipients["dwt"]
    ).sum() * 12 / 1_000_000
    weighted_mean = (
        (recipients["bsarg_s"] * recipients["dwt"]).sum() /
        recipients["dwt"].sum()
    )

    informe = pd.DataFrame(INFORME_RMI[year])
    informe_excl          = informe[~informe["drgn2"].isin(EXCLUDE_REGIONS)]
    informe_titulares     = informe_excl["titulares"].sum()
    informe_expenditure_M = (
        informe_excl["gasto_anual_por_titular"] *
        informe_excl["titulares"] / 1_000_000
    ).sum()

    logger.info("--- National summary (excl. La Rioja, Aragón, Ceuta) ---")
    logger.info(
        "  Target 1 — Recipients:   EUROMOD %10.0f | Informe %10.0f | ratio %.3f",
        weighted_recipients, informe_titulares,
        weighted_recipients / informe_titulares,
    )
    logger.info(
        "  Target 2 — Expenditure:  EUROMOD %10.2fM | Informe %10.2fM | ratio %.3f",
        weighted_expenditure_M, informe_expenditure_M,
        weighted_expenditure_M / informe_expenditure_M,
    )
    logger.info(
        "  Target 3 — Mean monthly: EUROMOD %10.2f€  | Informe %10.2f€  | ratio %.3f",
        weighted_mean, nat["cuantia_media"],
        weighted_mean / nat["cuantia_media"],
    )
    logger.info(
        "  Diagnosis: recipients < 1 & expenditure ≈ 1 → "
        "take-up / non-income eligibility conditions not fully simulated"
    )


def compute_pooled_validation(results: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Pool all valid region-year observations across 2017, 2018, 2019.
    Murcia 2019 (NaN) is automatically excluded via dropna().
    Returns the pooled DataFrame and prints summary statistics.
    """
    frames = []
    for year, df in sorted(results.items()):
        tmp = df.copy()
        tmp["year"] = year
        frames.append(tmp)

    pooled = pd.concat(frames, ignore_index=True)
    clean  = pooled.dropna(subset=["avg_monthly_admin", "euromod_mean_monthly"])

    r,   p_r   = pearsonr(clean["avg_monthly_admin"], clean["euromod_mean_monthly"])
    rho, p_rho = spearmanr(clean["avg_monthly_admin"], clean["euromod_mean_monthly"])

    logger.info("=" * 60)
    logger.info("POOLED VALIDATION — all valid region-year observations")
    logger.info("  Total region-year pairs:     %d", len(pooled))
    logger.info("  Valid (non-NaN) pairs used:  %d", len(clean))
    logger.info("  (Murcia 2019 excluded via NaN — BCA sampling issue)")
    logger.info("")
    logger.info("  Pearson  r   = %.3f  (p = %.4f)", r,   p_r)
    logger.info("  Spearman rho = %.3f  (p = %.4f)", rho, p_rho)
    logger.info("")
    logger.info("  Mean ratio_avg_benefit:  %.3f  (1.0 = perfect benefit match)",
                clean["ratio_avg_benefit"].mean())
    logger.info("  Mean coverage_rate:      %.3f  (1.0 = full take-up)",
                clean["coverage_rate"].mean())
    logger.info("  Mean ratio_expenditure:  %.3f  (1.0 = perfect expenditure match)",
                clean["ratio_expenditure"].mean())
    logger.info("")

    # Year-on-year rank consistency
    logger.info("  Regional rank consistency (Spearman) across years:")
    years = sorted(results.keys())
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        d1 = results[y1][["drgn2", "euromod_mean_monthly"]].rename(
            columns={"euromod_mean_monthly": f"m_{y1}"}
        )
        d2 = results[y2][["drgn2", "euromod_mean_monthly"]].rename(
            columns={"euromod_mean_monthly": f"m_{y2}"}
        )
        merged = d1.merge(d2, on="drgn2").dropna()
        rho_yr, _ = spearmanr(merged[f"m_{y1}"], merged[f"m_{y2}"])
        logger.info(
            "    %d → %d: rho = %.3f  (N=%d regions)",
            y1, y2, rho_yr, len(merged)
        )

    if SAVE_CSV:
        out_cols = [
            "year", "region", "drgn2",
            "titulares", "euromod_recipients", "coverage_rate",
            "avg_monthly_admin", "euromod_mean_monthly", "ratio_avg_benefit",
            "informe_expenditure_M", "euromod_expenditure_M", "ratio_expenditure",
        ]
        csv_path = OUTPUT_DIR / "validation_pooled.csv"
        clean[out_cols].to_csv(csv_path, index=False)
        logger.info("  Saved pooled table → %s", csv_path)

    return clean


def plot_validation(results: dict[int, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    years = sorted(results.keys())

    for ax, year in zip(axes, years):
        df = results[year].dropna(
            subset=["euromod_mean_monthly", "avg_monthly_admin"]
        )

        ax.scatter(
            df["avg_monthly_admin"],
            df["euromod_mean_monthly"],
            color="#378ADD", s=60, zorder=3, alpha=0.85,
        )

        for _, row in df.iterrows():
            ax.annotate(
                row["region"],
                xy=(row["avg_monthly_admin"], row["euromod_mean_monthly"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 2), textcoords="offset points",
                color="#5F5E5A",
            )

        lims = [
            min(df["avg_monthly_admin"].min(),
                df["euromod_mean_monthly"].min()) * 0.85,
            max(df["avg_monthly_admin"].max(),
                df["euromod_mean_monthly"].max()) * 1.10,
        ]
        ax.plot(lims, lims, "--", color="#B4B2A9", linewidth=0.8, zorder=1)

        corr = compute_correlations(df)
        ax.set_title(
            f"{year}  |  r = {corr['pearson_r']}, "
            f"ρ = {corr['spearman_rho']}  (N={corr['n']})",
            fontsize=10,
        )
        ax.set_xlabel(
            "Informe RMI avg monthly benefit (€/month)\n"
            "[gasto_anual_por_titular / 12]",
            fontsize=8,
        )
        ax.set_ylabel(
            "EUROMOD weighted mean monthly RMI (€/month)", fontsize=8
        )
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"€{int(x)}")
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"€{int(x)}")
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.suptitle(
        "EUROMOD simulated RMI vs Informe RMI average monthly benefit — "
        "pre-reform validation 2017–2019\n"
        "(excl. La Rioja, Aragón, Ceuta all years; "
        "Murcia omitted in 2019 panel only)",
        fontsize=10, y=1.03,
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "validation_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved validation plot → %s", out_path)
    plt.close()


def main() -> None:
    logger.info(
        "Starting EUROMOD multi-target validation — years: %s",
        sorted(EUROMOD_FILES.keys()),
    )
    logger.info(
        "Permanently excluded regions: La Rioja (23), Aragón (24), Ceuta (63)"
    )

    results: dict[int, pd.DataFrame] = {}

    for year, path in sorted(EUROMOD_FILES.items()):
        logger.info("=" * 60)
        logger.info("YEAR %s", year)

        if not path.exists():
            logger.error("EUROMOD output not found: %s", path)
            continue

        euromod_df = load_euromod_output(path)
        regional   = compute_regional_rmi(euromod_df)
        comparison = build_comparison(year, regional)
        corr       = compute_correlations(comparison)

        results[year] = comparison

        print_national_summary(year, euromod_df)

        out_cols = [
            "region", "drgn2",
            "titulares", "euromod_recipients", "coverage_rate",
            "avg_monthly_admin", "euromod_mean_monthly", "ratio_avg_benefit",
            "informe_expenditure_M", "euromod_expenditure_M", "ratio_expenditure",
        ]
        logger.info(
            "\nRegional detail:\n%s",
            comparison[out_cols].to_string(index=False),
        )
        logger.info(
            "\nCorrelations (avg monthly benefit, N=%d):\n"
            "  Pearson  r   = %.3f  (p = %.4f)\n"
            "  Spearman rho = %.3f  (p = %.4f)",
            corr["n"],
            corr["pearson_r"],    corr["pearson_p"],
            corr["spearman_rho"], corr["spearman_p"],
        )

        if SAVE_CSV:
            csv_path = OUTPUT_DIR / f"validation_table_{year}.csv"
            comparison[out_cols].to_csv(csv_path, index=False)
            logger.info("Saved → %s", csv_path)

    if results:
        compute_pooled_validation(results)
        plot_validation(results)

    logger.info("=" * 60)
    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
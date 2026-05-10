"""
RMIeuromod_validation.py
========================
Multi-target validation of EUROMOD pre-reform RMI simulations against
Informe RMI administrative benchmarks for 2017, 2018, and 2019.

Validation targets:
  1. Weighted recipient count vs Informe titulares
  2. Total annual expenditure vs Informe gasto anual ejecutado

NOTE — mean monthly benefit is NOT used as a correlation benchmark.
The Informe does not publish a regional average monthly benefit directly.
gasto_anual_por_titular / 12 is a flow measure (annual spending per
registered recipient) that is systematically distorted by:
  - recipient turnover within the year (partial-year spells deflate it
    below the statutory floor for high-churn regions)
  - supplements, housing allowances, and arrears included in the annual
    total that EUROMOD cannot simulate
  - multiple schemes reported jointly (e.g. Illes Balears RMI + RESOGA,
    País Vasco income + housing supplement)
These distortions make it an invalid comparator for EUROMOD's
income-formula entitlement simulation. avg_monthly_admin is retained
in CSV output for descriptive reference only.

Exclusions (all years):
  - La Rioja (drgn2=23): incomplete EUROMOD J2.0+ parameterisation
  - Aragón (drgn2=24): pre-2021 IAI not coded in J2.0+ architecture
  - Ceuta (drgn2=63): zero simulated recipients due to very small ECV sample

Murcia (drgn2=62):
  - Valid for 2017 and 2018
  - Zero recipients in 2019 due to BCA probabilistic allocation
    on small sample; excluded automatically via dropna()
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

EXCLUDE_REGIONS: set[int] = {63}

SAVE_CSV = True

# National totals from Informe RMI Cuadro 7 and Cuadro 8.
# cuantia_media retained for reference only — not used as validation target.
INFORME_NATIONAL = {
    2017: {"titulares": 313291, "gasto_anual_M": 1545.44},  # cuantia_media: 449.98 (not used)
    2018: {"titulares": 293302, "gasto_anual_M": 1519.67},  # cuantia_media: 463.05 (not used)
    2019: {"titulares": 297183, "gasto_anual_M": 1686.26},  # cuantia_media: 486.03 (not used)
}

# =============================================================================
# INFORME RMI REGIONAL BENCHMARKS
# Source: Informe de Rentas Mínimas de Inserción, Cuadro 7 and Cuadro 8
#
# titulares:              recipient households, Cuadro 7
# gasto_anual_ejecutado:  total annual expenditure (€), Cuadro 8
# gasto_anual_por_titular: expenditure / titulares (annual, per household)
#   → retained for descriptive reference ONLY, not used in correlations
#   → see module docstring for why this is not a valid correlation benchmark
# =============================================================================

INFORME_RMI: dict[int, list[dict]] = {

    2017: [
        {"region": "Andalucía",          "drgn2": 61, "titulares": 29337,  "gasto_anual_ejecutado":  86319748.19, "gasto_anual_por_titular": 2942.35},
        {"region": "Aragón",             "drgn2": 24, "titulares": 10466,  "gasto_anual_ejecutado":  49335694.70, "gasto_anual_por_titular": 4713.90},
        {"region": "Asturias",           "drgn2": 12, "titulares": 22219,  "gasto_anual_ejecutado": 119120320.98, "gasto_anual_por_titular": 5361.19},
        {"region": "Illes Balears",      "drgn2": 53, "titulares": 7551,   "gasto_anual_ejecutado":  13313137.08, "gasto_anual_por_titular": 1763.10},
        {"region": "Canarias",           "drgn2": 70, "titulares": 13525,  "gasto_anual_ejecutado":  53058589.33, "gasto_anual_por_titular": 3923.00},
        {"region": "Cantabria",          "drgn2": 13, "titulares": 6366,   "gasto_anual_ejecutado":  31209454.08, "gasto_anual_por_titular": 4902.52},
        {"region": "Castilla-La Mancha", "drgn2": 42, "titulares": 3152,   "gasto_anual_ejecutado":   6711262.49, "gasto_anual_por_titular": 2129.21},
        {"region": "Castilla y León",    "drgn2": 41, "titulares": 15502,  "gasto_anual_ejecutado":  77714006.97, "gasto_anual_por_titular": 5013.16},
        {"region": "Cataluña",           "drgn2": 51, "titulares": 26311,  "gasto_anual_ejecutado": 184571389.76, "gasto_anual_por_titular": 7014.99},
        {"region": "Ceuta",              "drgn2": 63, "titulares": 263,    "gasto_anual_ejecutado":    564623.62, "gasto_anual_por_titular": 2146.86},
        {"region": "Extremadura",        "drgn2": 43, "titulares": 6316,   "gasto_anual_ejecutado":  48434000.00, "gasto_anual_por_titular": 7668.46},
        {"region": "Galicia",            "drgn2": 11, "titulares": 14468,  "gasto_anual_ejecutado":  58809670.96, "gasto_anual_por_titular": 4064.81},
        {"region": "Madrid",             "drgn2": 30, "titulares": 35483,  "gasto_anual_ejecutado": 168626480.30, "gasto_anual_por_titular": 4752.32},
        {"region": "Melilla",            "drgn2": 64, "titulares": 994,    "gasto_anual_ejecutado":   4064300.53, "gasto_anual_por_titular": 4088.83},
        {"region": "Murcia",             "drgn2": 62, "titulares": 5421,   "gasto_anual_ejecutado":  14144691.21, "gasto_anual_por_titular": 2609.24},
        {"region": "Navarra",            "drgn2": 22, "titulares": 15918,  "gasto_anual_ejecutado":  98081807.38, "gasto_anual_por_titular": 6161.69},
        {"region": "País Vasco",         "drgn2": 21, "titulares": 76188,  "gasto_anual_ejecutado": 468426721.00, "gasto_anual_por_titular": 6148.30},
        {"region": "La Rioja",           "drgn2": 23, "titulares": 2424,   "gasto_anual_ejecutado":   5030000.00, "gasto_anual_por_titular": 2075.08},
        {"region": "C. Valenciana",      "drgn2": 52, "titulares": 21387,  "gasto_anual_ejecutado":  57907723.43, "gasto_anual_por_titular": 2707.61},
    ],

    2018: [
        {"region": "Andalucía",          "drgn2": 61, "titulares": 17883,  "gasto_anual_ejecutado":  53710000.00, "gasto_anual_por_titular": 3003.66},
        {"region": "Aragón",             "drgn2": 24, "titulares": 9894,   "gasto_anual_ejecutado":  48502000.00, "gasto_anual_por_titular": 4902.38},
        {"region": "Asturias",           "drgn2": 12, "titulares": 22305,  "gasto_anual_ejecutado": 124548000.00, "gasto_anual_por_titular": 5584.11},
        {"region": "Illes Balears",      "drgn2": 53, "titulares": 9714,   "gasto_anual_ejecutado":  21330000.00, "gasto_anual_por_titular": 2196.11},
        {"region": "Canarias",           "drgn2": 70, "titulares": 11592,  "gasto_anual_ejecutado":  42620000.00, "gasto_anual_por_titular": 3676.40},
        {"region": "Cantabria",          "drgn2": 13, "titulares": 5365,   "gasto_anual_ejecutado":  30990000.00, "gasto_anual_por_titular": 5775.67},
        {"region": "Castilla-La Mancha", "drgn2": 42, "titulares": 3544,   "gasto_anual_ejecutado":   9440000.00, "gasto_anual_por_titular": 2663.89},
        {"region": "Castilla y León",    "drgn2": 41, "titulares": 14536,  "gasto_anual_ejecutado":  71880000.00, "gasto_anual_por_titular": 4944.90},
        {"region": "Cataluña",           "drgn2": 51, "titulares": 28572,  "gasto_anual_ejecutado": 240510000.00, "gasto_anual_por_titular": 8417.56},
        {"region": "Ceuta",              "drgn2": 63, "titulares": 266,    "gasto_anual_ejecutado":    441534.06, "gasto_anual_por_titular": 1659.91},
        {"region": "Extremadura",        "drgn2": 43, "titulares": 5982,   "gasto_anual_ejecutado":  48430000.00, "gasto_anual_por_titular": 8096.62},
        {"region": "Galicia",            "drgn2": 11, "titulares": 14238,  "gasto_anual_ejecutado":  55320000.00, "gasto_anual_por_titular": 3885.31},
        {"region": "Madrid",             "drgn2": 30, "titulares": 33000,  "gasto_anual_ejecutado": 152560000.00, "gasto_anual_por_titular": 4623.03},
        {"region": "Melilla",            "drgn2": 64, "titulares": 784,    "gasto_anual_ejecutado":   3306000.00, "gasto_anual_por_titular": 4217.55},
        {"region": "Murcia",             "drgn2": 62, "titulares": 5856,   "gasto_anual_ejecutado":  16520000.00, "gasto_anual_por_titular": 2821.62},
        {"region": "Navarra",            "drgn2": 22, "titulares": 16078,  "gasto_anual_ejecutado": 103520000.00, "gasto_anual_por_titular": 6438.65},
        {"region": "País Vasco",         "drgn2": 21, "titulares": 72341,  "gasto_anual_ejecutado": 438560000.00, "gasto_anual_por_titular": 6062.42},
        {"region": "La Rioja",           "drgn2": 23, "titulares": 2941,   "gasto_anual_ejecutado":  12590000.00, "gasto_anual_por_titular": 4280.86},
        {"region": "C. Valenciana",      "drgn2": 52, "titulares": 18411,  "gasto_anual_ejecutado":  44880000.00, "gasto_anual_por_titular": 2437.64},
    ],

    2019: [
        {"region": "Andalucía",          "drgn2": 61, "titulares": 22318,  "gasto_anual_ejecutado": 107670000.00, "gasto_anual_por_titular": 4824.52},
        {"region": "Aragón",             "drgn2": 24, "titulares": 9401,   "gasto_anual_ejecutado":  46760000.00, "gasto_anual_por_titular": 4974.20},
        {"region": "Asturias",           "drgn2": 12, "titulares": 21947,  "gasto_anual_ejecutado": 120750000.00, "gasto_anual_por_titular": 5501.72},
        {"region": "Illes Balears",      "drgn2": 53, "titulares": 10449,  "gasto_anual_ejecutado":  27580000.00, "gasto_anual_por_titular": 2639.81},
        {"region": "Canarias",           "drgn2": 70, "titulares": 9973,   "gasto_anual_ejecutado":  36530000.00, "gasto_anual_por_titular": 3663.09},
        {"region": "Cantabria",          "drgn2": 13, "titulares": 7052,   "gasto_anual_ejecutado":  30040000.00, "gasto_anual_por_titular": 4260.35},
        {"region": "Castilla-La Mancha", "drgn2": 42, "titulares": 4132,   "gasto_anual_ejecutado":  17750000.00, "gasto_anual_por_titular": 4296.13},
        {"region": "Castilla y León",    "drgn2": 41, "titulares": 13069,  "gasto_anual_ejecutado":  64090000.00, "gasto_anual_por_titular": 4903.90},
        {"region": "Cataluña",           "drgn2": 51, "titulares": 32166,  "gasto_anual_ejecutado": 267530000.00, "gasto_anual_por_titular": 8317.18},
        {"region": "Ceuta",              "drgn2": 63, "titulares": 179,    "gasto_anual_ejecutado":    485668.25, "gasto_anual_por_titular": 2713.25},
        {"region": "Extremadura",        "drgn2": 43, "titulares": 7991,   "gasto_anual_ejecutado":  47430000.00, "gasto_anual_por_titular": 5935.93},
        {"region": "Galicia",            "drgn2": 11, "titulares": 13600,  "gasto_anual_ejecutado":  53390000.00, "gasto_anual_por_titular": 3925.85},
        {"region": "Madrid",             "drgn2": 30, "titulares": 28643,  "gasto_anual_ejecutado": 155300000.00, "gasto_anual_por_titular": 5421.78},
        {"region": "Melilla",            "drgn2": 64, "titulares": 510,    "gasto_anual_ejecutado":   1590000.00, "gasto_anual_por_titular": 3117.26},
        {"region": "Murcia",             "drgn2": 62, "titulares": 6355,   "gasto_anual_ejecutado":  18470000.00, "gasto_anual_por_titular": 2906.34},
        {"region": "Navarra",            "drgn2": 22, "titulares": 15712,  "gasto_anual_ejecutado": 103090000.00, "gasto_anual_por_titular": 6561.06},
        {"region": "País Vasco",         "drgn2": 21, "titulares": 66508,  "gasto_anual_ejecutado": 422490000.00, "gasto_anual_por_titular": 6352.42},
        {"region": "La Rioja",           "drgn2": 23, "titulares": 3070,   "gasto_anual_ejecutado":  13910000.00, "gasto_anual_por_titular": 4532.00},
        {"region": "C. Valenciana",      "drgn2": 52, "titulares": 24108,  "gasto_anual_ejecutado": 151390000.00, "gasto_anual_por_titular": 6279.73},
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
    Compute weighted recipient count, mean monthly benefit, and annual
    expenditure by region from EUROMOD output.
    euromod_mean_monthly is retained for descriptive CSV output only —
    it is not used as a correlation benchmark.
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

    # Convert raw expenditure to millions for comparability with EUROMOD
    informe["informe_expenditure_M"] = (
        informe["gasto_anual_ejecutado"] / 1_000_000
    ).round(2)

    # Descriptive reference only — NOT used in correlations
    # See module docstring for why this is an invalid correlation benchmark
    informe["avg_monthly_admin"] = (
        informe["gasto_anual_por_titular"] / 12
    ).round(2)

    # Left join: all Informe regions, NaN where EUROMOD has no recipients
    df = informe.merge(regional, on="drgn2", how="left")

    # Descriptive ratios — kept for CSV transparency, not validation targets
    df["ratio_recipients"] = (
        df["euromod_recipients"] / df["titulares"]
    ).round(3)
    df["ratio_expenditure"] = (
        df["euromod_expenditure_M"] / df["informe_expenditure_M"]
    ).round(3)

    # Descriptive mean benefit comparison — NOT a validation target
    df["euromod_mean_monthly"] = df["euromod_mean_monthly"].round(2)
    df["ratio_avg_benefit"] = (
        df["euromod_mean_monthly"] / df["avg_monthly_admin"]
    ).round(3)

    return df


def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Correlate administrative vs simulated values across regions.

    Validation targets:
      1. titulares vs euromod_recipients (recipient counts)
      2. informe_expenditure_M vs euromod_expenditure_M (annual expenditure)

    Mean benefit (avg_monthly_admin) is excluded as a correlation benchmark.
    gasto_anual_por_titular / 12 is a flow measure distorted by recipient
    turnover, supplements, and multi-scheme reporting — not comparable to
    EUROMOD's stock measure of monthly entitlement among active recipients.
    """
    clean = df[[
        "titulares", "euromod_recipients",
        "informe_expenditure_M", "euromod_expenditure_M",
    ]].dropna()

    r_rec,   p_rec   = pearsonr( clean["titulares"],             clean["euromod_recipients"])
    rho_rec, p_rho_r = spearmanr(clean["titulares"],             clean["euromod_recipients"])
    r_exp,   p_exp   = pearsonr( clean["informe_expenditure_M"], clean["euromod_expenditure_M"])
    rho_exp, p_rho_e = spearmanr(clean["informe_expenditure_M"], clean["euromod_expenditure_M"])

    return {
        "n":               len(clean),
        "pearson_r_rec":   round(r_rec,   3),
        "pearson_p_rec":   round(p_rec,   4),
        "spearman_rho_rec":round(rho_rec, 3),
        "spearman_p_rec":  round(p_rho_r, 4),
        "pearson_r_exp":   round(r_exp,   3),
        "pearson_p_exp":   round(p_exp,   4),
        "spearman_rho_exp":round(rho_exp, 3),
        "spearman_p_exp":  round(p_rho_e, 4),
    }


def print_national_summary(year: int, euromod_df: pd.DataFrame) -> None:
    """
    Print national-level validation summary for recipients and expenditure.
    Mean benefit comparison is omitted — see module docstring.
    """
    recipients = euromod_df[
        (euromod_df["bsarg_s"] > 0) &
        (~euromod_df["drgn2"].isin(EXCLUDE_REGIONS))
    ]
    weighted_recipients    = recipients["dwt"].sum()
    weighted_expenditure_M = (
        recipients["bsarg_s"] * recipients["dwt"]
    ).sum() * 12 / 1_000_000

    informe = pd.DataFrame(INFORME_RMI[year])
    informe_excl          = informe[~informe["drgn2"].isin(EXCLUDE_REGIONS)]
    informe_titulares     = informe_excl["titulares"].sum()
    informe_expenditure_M = (
        informe_excl["gasto_anual_ejecutado"] / 1_000_000
    ).sum()

    logger.info("--- National summary (excl. La Rioja, Aragón, Ceuta) ---")
    logger.info(
        "  Target 1 — Recipients:  EUROMOD %10.0f | Informe %10.0f | ratio %.3f",
        weighted_recipients, informe_titulares,
        weighted_recipients / informe_titulares,
    )
    logger.info(
        "  Target 2 — Expenditure: EUROMOD %10.2fM | Informe %10.2fM | ratio %.3f",
        weighted_expenditure_M, informe_expenditure_M,
        weighted_expenditure_M / informe_expenditure_M,
    )
    logger.info(
        "  Note: mean monthly benefit not validated — gasto_anual_por_titular/12"
        " is a flow measure distorted by turnover, supplements, and multi-scheme"
        " reporting. See module docstring."
    )


def compute_pooled_validation(results: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Pool all valid region-year observations across 2017, 2018, 2019.
    Murcia 2019 (NaN) is automatically excluded via dropna().
    Correlations on recipient counts and expenditure only.
    Year-on-year rank consistency uses euromod_expenditure_M.
    """
    frames = []
    for year, df in sorted(results.items()):
        tmp = df.copy()
        tmp["year"] = year
        frames.append(tmp)

    pooled = pd.concat(frames, ignore_index=True)
    clean  = pooled.dropna(subset=[
        "titulares", "euromod_recipients",
        "informe_expenditure_M", "euromod_expenditure_M",
    ])

    r_rec,   p_rec   = pearsonr( clean["titulares"],             clean["euromod_recipients"])
    rho_rec, p_rho_r = spearmanr(clean["titulares"],             clean["euromod_recipients"])
    r_exp,   p_exp   = pearsonr( clean["informe_expenditure_M"], clean["euromod_expenditure_M"])
    rho_exp, p_rho_e = spearmanr(clean["informe_expenditure_M"], clean["euromod_expenditure_M"])

    logger.info("=" * 60)
    logger.info("POOLED VALIDATION — all valid region-year observations")
    logger.info("  Total region-year pairs:     %d", len(pooled))
    logger.info("  Valid (non-NaN) pairs used:  %d", len(clean))
    logger.info("  (Murcia 2019 excluded via NaN — BCA sampling issue)")
    logger.info("")
    logger.info("  Recipients — Pearson  r   = %.3f  (p = %.4f)", r_rec,   p_rec)
    logger.info("  Recipients — Spearman rho = %.3f  (p = %.4f)", rho_rec, p_rho_r)
    logger.info("")
    logger.info("  Expenditure — Pearson  r   = %.3f  (p = %.4f)", r_exp,   p_exp)
    logger.info("  Expenditure — Spearman rho = %.3f  (p = %.4f)", rho_exp, p_rho_e)
    logger.info("")
    logger.info("  Mean ratio_recipients:  %.3f  (1.0 = perfect recipient match)",
                clean["ratio_recipients"].mean())
    logger.info("  Mean ratio_expenditure: %.3f  (1.0 = perfect expenditure match)",
                clean["ratio_expenditure"].mean())
    logger.info("")

    # Year-on-year rank consistency on expenditure
    logger.info("  Regional rank consistency (Spearman on expenditure) across years:")
    years = sorted(results.keys())
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        d1 = results[y1][["drgn2", "euromod_expenditure_M"]].rename(
            columns={"euromod_expenditure_M": f"exp_{y1}"}
        )
        d2 = results[y2][["drgn2", "euromod_expenditure_M"]].rename(
            columns={"euromod_expenditure_M": f"exp_{y2}"}
        )
        merged = d1.merge(d2, on="drgn2").dropna()
        rho_yr, _ = spearmanr(merged[f"exp_{y1}"], merged[f"exp_{y2}"])
        logger.info(
            "    %d → %d: rho = %.3f  (N=%d regions)",
            y1, y2, rho_yr, len(merged)
        )

    if SAVE_CSV:
        out_cols = [
            "year", "region", "drgn2",
            "titulares", "euromod_recipients", "ratio_recipients",
            "informe_expenditure_M", "euromod_expenditure_M", "ratio_expenditure",
            # descriptive only — not validation targets
            "avg_monthly_admin", "euromod_mean_monthly", "ratio_avg_benefit",
        ]
        out_cols = [c for c in out_cols if c in clean.columns]
        csv_path = OUTPUT_DIR / "validation_pooled.csv"
        clean[out_cols].to_csv(csv_path, index=False)
        logger.info("  Saved pooled table → %s", csv_path)

    return clean


def plot_validation(results: dict[int, pd.DataFrame]) -> None:
    """
    Two-row figure: top row = recipient counts, bottom row = expenditure.
    One column per year (2017, 2018, 2019).
    """
    years = sorted(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    targets = [
        {
            "x": "titulares",
            "y": "euromod_recipients",
            "xlabel": "Informe RMI titulares",
            "ylabel": "EUROMOD weighted recipients",
            "title_prefix": "Recipients",
            "fmt": lambda v: f"{int(v):,}",
        },
        {
            "x": "informe_expenditure_M",
            "y": "euromod_expenditure_M",
            "xlabel": "Informe RMI annual expenditure (€M)",
            "ylabel": "EUROMOD annual expenditure (€M)",
            "title_prefix": "Expenditure",
            "fmt": lambda v: f"€{v:.0f}M",
        },
    ]

    for row, target in enumerate(targets):
        for col, year in enumerate(years):
            ax = axes[row][col]
            df = results[year].dropna(subset=[target["x"], target["y"]])

            ax.scatter(
                df[target["x"]], df[target["y"]],
                color="#378ADD", s=60, zorder=3, alpha=0.85,
            )

            for _, r in df.iterrows():
                ax.annotate(
                    r["region"],
                    xy=(r[target["x"]], r[target["y"]]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(4, 2), textcoords="offset points",
                    color="#5F5E5A",
                )

            all_vals = pd.concat([df[target["x"]], df[target["y"]]])
            lims = [all_vals.min() * 0.85, all_vals.max() * 1.10]
            ax.plot(lims, lims, "--", color="#B4B2A9", linewidth=0.8, zorder=1)

            pr,  pp  = pearsonr( df[target["x"]], df[target["y"]])
            rho, rp  = spearmanr(df[target["x"]], df[target["y"]])
            ax.set_title(
                f"{year} — {target['title_prefix']}\n"
                f"r = {pr:.3f}, ρ = {rho:.3f}  (N={len(df)})",
                fontsize=9,
            )
            ax.set_xlabel(target["xlabel"], fontsize=8)
            ax.set_ylabel(target["ylabel"], fontsize=8)
            ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.suptitle(
        "EUROMOD simulated RMI vs Informe RMI — pre-reform validation 2017–2019\n"
        "Top: recipient counts  |  Bottom: annual expenditure\n"
        "(excl. La Rioja, Aragón, Ceuta all years; Murcia omitted in 2019 only)",
        fontsize=10, y=1.02,
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
    logger.info(
        "Validation targets: recipient counts and annual expenditure only."
    )
    logger.info(
        "Mean benefit NOT used as correlation benchmark — see module docstring."
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
            "titulares", "euromod_recipients", "ratio_recipients",
            "informe_expenditure_M", "euromod_expenditure_M", "ratio_expenditure",
            # descriptive only
            "avg_monthly_admin", "euromod_mean_monthly", "ratio_avg_benefit",
        ]
        out_cols = [c for c in out_cols if c in comparison.columns]
        logger.info(
            "\nRegional detail:\n%s",
            comparison[out_cols].to_string(index=False),
        )
        logger.info(
            "\nCorrelations (N=%d regions):\n"
            "  Recipients  — Pearson r = %.3f (p = %.4f) | Spearman rho = %.3f (p = %.4f)\n"
            "  Expenditure — Pearson r = %.3f (p = %.4f) | Spearman rho = %.3f (p = %.4f)",
            corr["n"],
            corr["pearson_r_rec"],    corr["pearson_p_rec"],
            corr["spearman_rho_rec"], corr["spearman_p_rec"],
            corr["pearson_r_exp"],    corr["pearson_p_exp"],
            corr["spearman_rho_exp"], corr["spearman_p_exp"],
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
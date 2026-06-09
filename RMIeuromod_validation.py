"""
RMIeuromod_validation.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from src.constants import INFORME_RMI

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

EXCLUDE_REGIONS: set[int] = {23, 24, 63, 64}

SAVE_CSV = True

# National totals from Informe RMI Cuadro 7 and Cuadro 8.
# cuantia_media retained for reference only — not used as validation target.
INFORME_NATIONAL = {
    2017: {"titulares": 313291, "gasto_anual_M": 1545.44},  # cuantia_media: 449.98 (not used)
    2018: {"titulares": 293302, "gasto_anual_M": 1519.67},  # cuantia_media: 463.05 (not used)
    2019: {"titulares": 297183, "gasto_anual_M": 1686.26},  # cuantia_media: 486.03 (not used)
}


def load_euromod_output(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )
    return df

def compute_regional_rmi(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Compute weighted recipient count, mean monthly benefit, and annual
    expenditure by region from EUROMOD output.

    bsarg_s is normally assigned to one person per recipient household, so
    positive person rows are treated as simulated claimant units. Rare duplicate
    positive rows within the same household are collapsed to avoid double-counting.
    """
    recipients = recipient_units_from_person_output(df, year)

    regional = (
        recipients.groupby("drgn2")
        .apply(lambda x: pd.Series({
            "euromod_recipients": x["dwt"].sum(),
            "euromod_mean_monthly": (
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

    informe["avg_monthly_admin"] = (
        informe["gasto_anual_por_titular"] / 12
    ).round(2)

    df = informe.merge(regional, on="drgn2", how="left")

    df["ratio_recipients"] = (
        df["euromod_recipients"] / df["titulares"]
    ).round(3)
    df["ratio_expenditure"] = (
        df["euromod_expenditure_M"] / df["informe_expenditure_M"]
    ).round(3)

    df["euromod_mean_monthly"] = df["euromod_mean_monthly"].round(2)
    df["ratio_avg_benefit"] = (
        df["euromod_mean_monthly"] / df["avg_monthly_admin"]
    ).round(3)

    return df


def compute_correlations(df: pd.DataFrame) -> dict:
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
    recipients = recipient_units_from_person_output(euromod_df, year)

    weighted_recipients = recipients["dwt"].sum()
    weighted_expenditure_M = (
        recipients["bsarg_s"] * recipients["dwt"]
    ).sum() * 12 / 1_000_000

    informe = pd.DataFrame(INFORME_RMI[year])
    informe_excl = informe[~informe["drgn2"].isin(EXCLUDE_REGIONS)]
    informe_titulares = informe_excl["titulares"].sum()
    informe_expenditure_M = (
        informe_excl["gasto_anual_ejecutado"] / 1_000_000
    ).sum()

    logger.info("--- National summary (excl. Ceuta and Melilla) ---")
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
            "avg_monthly_admin", "euromod_mean_monthly", "ratio_avg_benefit",
        ]
        out_cols = [c for c in out_cols if c in clean.columns]
        csv_path = OUTPUT_DIR / "validation_pooled.csv"
        clean[out_cols].to_csv(csv_path, index=False)
        logger.info("  Saved pooled table → %s", csv_path)

    return clean


def plot_validation(results: dict[int, pd.DataFrame]) -> None:
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
        "(excl. Melilla and Ceuta all years; Murcia omitted in 2019 only)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "validation_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved validation plot → %s", out_path)
    plt.close()

def recipient_units_from_person_output(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    EUROMOD output is person-level. For bsarg_s, diagnostics show that the
    simulated RMI is normally assigned to one person per recipient household.
    Therefore, validation against administrative titulares uses positive
    bsarg_s person rows as simulated claimant units.

    Rare households with multiple positive bsarg_s rows are collapsed to one
    unit to avoid double-counting.
    """
    required = {"idhh", "idperson", "drgn2", "dwt", "bsarg_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Year {year}: missing required columns: {sorted(missing)}")

    pos = df[
        (df["bsarg_s"] > 0) &
        (~df["drgn2"].isin(EXCLUDE_REGIONS))
    ].copy()

    diagnostic = (
        pos.groupby("idhh")
        .agg(
            n_pos_rows=("idperson", "size"),
            n_unique_amounts=("bsarg_s", "nunique"),
            drgn2_nunique=("drgn2", "nunique"),
            dwt_nunique=("dwt", "nunique"),
        )
        .reset_index()
    )

    logger.info("Year %s: positive bsarg_s person rows: %d", year, len(pos))
    logger.info("Year %s: positive bsarg_s households: %d", year, pos["idhh"].nunique())

    if not diagnostic.empty:
        logger.info(
            "Year %s: mean positive rows per recipient household: %.3f",
            year,
            diagnostic["n_pos_rows"].mean()
        )
        logger.info(
            "Year %s: households with >1 positive bsarg_s row: %d",
            year,
            int((diagnostic["n_pos_rows"] > 1).sum())
        )
        logger.info(
            "Year %s: households with multiple positive bsarg_s amounts: %d",
            year,
            int((diagnostic["n_unique_amounts"] > 1).sum())
        )

    if (diagnostic["drgn2_nunique"] > 1).any():
        raise ValueError(f"Year {year}: drgn2 varies within recipient household")

    if (diagnostic["dwt_nunique"] > 1).any():
        logger.warning("Year %s: dwt varies within some recipient households", year)

    units = (
        pos.groupby("idhh", as_index=False)
        .agg(
            drgn2=("drgn2", "first"),
            dwt=("dwt", "first"),
            bsarg_s=("bsarg_s", "max"),
            n_pos_rows=("idperson", "size"),
            n_unique_amounts=("bsarg_s", "nunique"),
        )
    )

    duplicated = units[units["n_pos_rows"] > 1]
    if len(duplicated) > 0:
        logger.warning(
            "Year %s: collapsed %d households with multiple positive bsarg_s rows "
            "using max(bsarg_s). Inspect these cases.",
            year, len(duplicated)
        )

    return units


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
        regional = compute_regional_rmi(euromod_df, year)
        comparison = build_comparison(year, regional)
        corr       = compute_correlations(comparison)

        results[year] = comparison

        print_national_summary(year, euromod_df)

        out_cols = [
            "region", "drgn2",
            "titulares", "euromod_recipients", "ratio_recipients",
            "informe_expenditure_M", "euromod_expenditure_M", "ratio_expenditure",
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
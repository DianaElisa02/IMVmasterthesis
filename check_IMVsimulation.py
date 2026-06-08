"""
check_IMVsimulation.py
======================
Diagnostic checks for EUROMOD IMV counterfactual simulation outputs.

Simulation logic (from EUROMOD Spain country report)
-----------------------------------------------------
  bsa00_s  — national IMV (Ingreso Mínimo Vital), simulated FIRST
  bsarg_s  — regional RMI (Rentas Mínimas de Inserción), simulated SECOND,
              with bsa00_s already counted in the household income test so the
              regional scheme only tops up if income + IMV < regional threshold

Total post-reform protection = bsa00_s + bsarg_s
Exposure gain per household  = (bsa00_s + bsarg_s) - bsarg_s_prereform

------------------
La Rioja  (drgn2=23): Renta de Ciudadanía — coded in 2022 EUROMOD with
  plausible amounts (€463–723/month). The €1 bug was only in the pre-reform
  STD files where $IPREM resolved incorrectly. Must verify it does NOT
  carry over to these IMV counterfactual files.

Aragón (drgn2=24): Prestación Aragonesa Complementaria del IMV — from 2022
  this scheme follows the SAME rules as the national IMV and is explicitly
  a complement to it. It replaced the old Ingreso Aragonés de Inserción
  (which ended October 2021). Must verify bsarg_s is plausible here.

Structure
---------
1. National summary     — totals for bsa00_s, bsarg_s, and combined
2. Implausible value checks — negatives, age (IMV: dag ≥ 23, or 18 with
                              children), employed recipients, zero weights
3. La Rioja / Aragón deep-dive — bsa00_s AND bsarg_s separately, checks
                              whether the €1 bug persists in these files
4. Benefit distribution — regional table for both variables and their sum
"""

from __future__ import annotations

import pandas as pd
import numpy as np

IMV_STATUTORY_SINGLE_2022 = 547.0   # €/month (weighted avg Jan–Dec 2022)

# Plausible range for mean monthly bsa00_s (IMV national, €/month)
# Floor: small top-up for households just below threshold
# Ceiling: 220% of basic amount (statutory cap for large households)
IMV_MIN_PLAUSIBLE  = 10.0
IMV_MAX_PLAUSIBLE  = 1300.0

# Plausible range for mean monthly bsarg_s (regional RMI, €/month)
# Regionally coded; País Vasco goes up to €1,250+, most others €400–800
RMI_MIN_PLAUSIBLE  = 10.0
RMI_MAX_PLAUSIBLE  = 1400.0

# Threshold below which a mean monthly benefit is flagged as a placeholder
PLACEHOLDER_THRESHOLD = 5.0   # €/month

# Exclude Ceuta  and Melillafrom national summaries (small population, different dynamic)
EXCLUDE_FROM_NATIONAL = {63, 64}

FILES = {
    2017: "/workspaces/IMVmasterthesis/input_data/euromod_output/IMV_2022ruleson2017.txt",
    2018: "/workspaces/IMVmasterthesis/input_data/euromod_output/IMV_2022ruleson2018.txt",
    2019: "/workspaces/IMVmasterthesis/input_data/euromod_output/IMV_2022ruleson2019.txt",
}

REGION_NAMES = {
    11: "Galicia",            12: "Asturias",          13: "Cantabria",
    21: "País Vasco",         22: "Navarra",            23: "La Rioja",
    24: "Aragón",             30: "Madrid",             41: "Castilla y León",
    42: "Castilla-La Mancha", 43: "Extremadura",        51: "Cataluña",
    52: "C. Valenciana",      53: "Illes Balears",      61: "Andalucía",
    62: "Murcia",             63: "Ceuta",              64: "Melilla",
    70: "Canarias",
}

def load_euromod_output(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False).str.strip(),
            errors="coerce",
        )
    return df

def sep(title: str = "") -> None:
    if title:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
    else:
        print()


def flag(condition: bool, msg_ok: str, msg_fail: str) -> str:
    return f"  {'✓' if condition else '✗'}  {msg_ok if condition else msg_fail}"


def regional_summary(df: pd.DataFrame, exclude: set[int] | None = None) -> pd.DataFrame:
    mask = (df["bsa00_s"] > 0) | (df["bsarg_s"] > 0)
    if exclude:
        mask &= ~df["drgn2"].isin(exclude)
    hh = df[mask].copy()
    hh["total_s"] = hh["bsa00_s"].fillna(0) + hh["bsarg_s"].fillna(0)

    rows = []
    for drgn2, grp in hh.groupby("drgn2"):
        w = grp["dwt"].sum()

        def wmean(col):
            rec = grp[grp[col] > 0]
            return (rec[col] * rec["dwt"]).sum() / rec["dwt"].sum() if rec["dwt"].sum() > 0 else 0.0

        def wexp(col):
            return (grp[col].fillna(0) * grp["dwt"]).sum() * 12 / 1_000_000

        rows.append({
            "drgn2":           int(drgn2),
            "region":          REGION_NAMES.get(int(drgn2), f"Region {int(drgn2)}"),
            "w_rec_imv":       round(grp[grp["bsa00_s"] > 0]["dwt"].sum(), 0),
            "mean_imv":        round(wmean("bsa00_s"), 2),
            "exp_imv_M":       round(wexp("bsa00_s"), 3),
            "w_rec_rmi":       round(grp[grp["bsarg_s"] > 0]["dwt"].sum(), 0),
            "mean_rmi":        round(wmean("bsarg_s"), 2),
            "exp_rmi_M":       round(wexp("bsarg_s"), 3),
            "exp_total_M":     round(wexp("total_s"), 3),
        })
    return pd.DataFrame(rows).sort_values("drgn2").reset_index(drop=True)

def print_national_summary(year: int, df: pd.DataFrame) -> None:
    sep("SECTION 1 — National summary  (excl. Ceuta)")

    d = df[~df["drgn2"].isin(EXCLUDE_FROM_NATIONAL)].copy()
    d["total_s"] = d["bsa00_s"].fillna(0) + d["bsarg_s"].fillna(0)

    def wsum(col): return (d[col].fillna(0) * d["dwt"]).sum()
    def wmean(col):
        rec = d[d[col] > 0]
        return (rec[col] * rec["dwt"]).sum() / rec["dwt"].sum() if rec["dwt"].sum() > 0 else 0.0

    w_imv  = d[d["bsa00_s"] > 0]["dwt"].sum()
    w_rmi  = d[d["bsarg_s"] > 0]["dwt"].sum()
    exp_imv  = wsum("bsa00_s") * 12 / 1_000_000
    exp_rmi  = wsum("bsarg_s") * 12 / 1_000_000
    exp_tot  = wsum("total_s") * 12 / 1_000_000

    print(f"  NOTE: 2022 rules applied to {year} ECV households — no Informe benchmark.")
    print(f"  bsa00_s = national IMV  |  bsarg_s = regional RMI  |  total = sum")
    print()
    print(f"  {'Variable':<12} {'W.recipients':>14} {'Mean €/mo':>10} {'Ann.exp M€':>12}")
    print(f"  {'─'*12} {'─'*14} {'─'*10} {'─'*12}")
    print(f"  {'bsa00_s':<12} {w_imv:>14,.0f} {wmean('bsa00_s'):>10.2f} {exp_imv:>12.2f}")
    print(f"  {'bsarg_s':<12} {w_rmi:>14,.0f} {wmean('bsarg_s'):>10.2f} {exp_rmi:>12.2f}")
    print(f"  {'total':<12} {'':>14} {'':>10} {exp_tot:>12.2f}")
    print()

    ratio = wmean("bsa00_s") / IMV_STATUTORY_SINGLE_2022 if wmean("bsa00_s") > 0 else 0
    print(f"  IMV statutory single-adult 2022: €{IMV_STATUTORY_SINGLE_2022:.0f}/mo")
    print(f"  Mean bsa00_s / statutory:        {ratio:.3f}  "
          f"({'plausible' if 0.3 <= ratio <= 2.5 else '← CHECK'})")

def print_implausible_checks(df: pd.DataFrame) -> None:
    sep("SECTION 2 — Implausible value checks")

    rec = df[df["bsa00_s"] > 0].copy()

    neg = (rec["bsa00_s"] < 0).sum()
    print(flag(neg == 0,
               "No negative bsa00_s values",
               f"{neg} negative bsa00_s values detected"))

    # Underage recipients — IMV requires dag ≥ 23 (or 18 with dependent children)
    if "dag" in rec.columns:
        under23 = rec[rec["dag"] < 23]
        w_under23 = under23["dwt"].sum()
        pct_under23 = 100 * w_under23 / rec["dwt"].sum() if rec["dwt"].sum() > 0 else 0
        # Some under-23 are valid (18–22 with children) — flag only if share is high
        print(f"  ~  Recipients dag < 23 (valid if children present): "
              f"{len(under23)} obs  (weighted: {w_under23:,.0f}, {pct_under23:.1f}%) "
              f"{'← review if > 5%' if pct_under23 > 5 else '✓'}")

        under18 = rec[rec["dag"] < 18]
        w_under18 = under18["dwt"].sum()
        print(flag(w_under18 == 0,
                   "No recipients dag < 18 (never valid under IMV)",
                   f"Recipients dag < 18: {len(under18)} obs  (weighted: {w_under18:,.0f}) ← ERROR"))

        # Over-65: IMV has no age ceiling, but high share is unexpected
        overage = rec[rec["dag"] > 65]
        w_overage = overage["dwt"].sum()
        pct_old = 100 * w_overage / rec["dwt"].sum() if rec["dwt"].sum() > 0 else 0
        print(f"  ~  Recipients dag > 65: {len(overage)} obs  "
              f"(weighted: {w_overage:,.0f}, {pct_old:.1f}%) "
              f"{'← review if > 10%' if pct_old > 10 else '✓'}")
    else:
        print("  !  dag column not found — age checks skipped")

    # Employed recipients (les == 3): IMV allows top-up for working poor
    if "les" in rec.columns:
        employed = rec[rec["les"] == 3]
        w_emp = employed["dwt"].sum()
        pct_emp = 100 * w_emp / rec["dwt"].sum() if rec["dwt"].sum() > 0 else 0
        print(f"  ~  Recipients les=3 (employed/working poor): {len(employed)} obs  "
              f"(weighted: {w_emp:,.0f}, {pct_emp:.1f}%) "
              f"{'← unusually high, review' if pct_emp > 40 else '✓'}")
    else:
        print("  !  les column not found — employment status check skipped")

    # Zero-weight records
    zero_wt = (df["dwt"] <= 0).sum()
    print(flag(zero_wt == 0,
               "No zero or negative survey weights",
               f"{zero_wt} records with dwt ≤ 0"))

    # Missing drgn2
    miss_region = df["drgn2"].isna().sum()
    print(flag(miss_region == 0,
               "No missing drgn2 values",
               f"{miss_region} records with missing drgn2"))

def print_broken_region_diagnostics(year: int, df: pd.DataFrame) -> None:
    sep("SECTION 3 — La Rioja (23) & Aragón (24) deep-dive")
    print("  Checking bsa00_s (national IMV) AND bsarg_s (regional RMI) separately.")
    print("  The €1 bug was in the pre-reform STD files. Question: does it persist here?")
    print("  For Aragón: from 2022 bsarg_s = Prestación Aragonesa Complementaria")
    print("  (same rules as national IMV, explicitly a complement to it).")

    for drgn2, name in [(23, "La Rioja"), (24, "Aragón")]:
        print(f"\n  ── {name}  (drgn2={drgn2}) ──")

        reg = df[df["drgn2"] == drgn2].copy()
        if reg.empty:
            print("    No observations in EUROMOD output.")
            continue

        total_pop = reg["dwt"].sum()
        print(f"    Weighted population: {total_pop:,.0f}")

        for var, label in [("bsa00_s", "national IMV"), ("bsarg_s", "regional RMI")]:
            rec   = reg[reg[var] > 0]
            w_rec = rec["dwt"].sum()
            pct   = 100 * w_rec / total_pop if total_pop > 0 else 0
            print(f"\n    [{var} — {label}]")
            print(f"      Recipients (weighted): {w_rec:>12,.0f}  ({pct:.1f}% of region pop)")

            if rec.empty:
                print(f"      No recipients — {var} produces zero entitlements")
                continue

            b = rec[var]
            print(f"      min:    {b.min():>10.2f} €/mo")
            print(f"      p25:    {b.quantile(0.25):>10.2f} €/mo")
            print(f"      median: {b.median():>10.2f} €/mo")
            print(f"      p75:    {b.quantile(0.75):>10.2f} €/mo")
            print(f"      max:    {b.max():>10.2f} €/mo")

            unique_vals = b.round(2).unique()
            if len(unique_vals) == 1 and unique_vals[0] <= PLACEHOLDER_THRESHOLD:
                print(f"      *** ALL values = €{unique_vals[0]:.2f} — PLACEHOLDER/BUG CONFIRMED ***")
            elif b.max() <= PLACEHOLDER_THRESHOLD:
                print(f"      *** All values ≤ €{PLACEHOLDER_THRESHOLD:.2f} — effective placeholder ***")
            else:
                print(f"      Benefit varies ({len(unique_vals)} unique values) — not a uniform placeholder  ✓")

            w_mean = (rec[var] * rec["dwt"]).sum() / w_rec
            exp_M  = (rec[var] * rec["dwt"]).sum() * 12 / 1_000_000
            print(f"      Weighted mean: {w_mean:.2f} €/mo  |  Simulated exp: {exp_M:.3f} M€/yr")

        reg["total_s"] = reg["bsa00_s"].fillna(0) + reg["bsarg_s"].fillna(0)
        any_rec = reg[reg["total_s"] > 0]
        w_any   = any_rec["dwt"].sum()
        exp_tot = (reg["total_s"] * reg["dwt"]).sum() * 12 / 1_000_000
        print(f"\n    [TOTAL bsa00_s + bsarg_s]")
        print(f"      Recipients receiving either benefit: {w_any:,.0f}")
        print(f"      Combined simulated expenditure:     {exp_tot:.3f} M€/yr")

def print_benefit_distribution(year: int, df: pd.DataFrame) -> None:
    sep("SECTION 4 — Regional benefit distribution (all regions)")

    reg = regional_summary(df)

    imv_placeholders = reg[reg["mean_imv"].between(0.01, PLACEHOLDER_THRESHOLD)]
    rmi_placeholders = reg[reg["mean_rmi"].between(0.01, PLACEHOLDER_THRESHOLD)]

    if imv_placeholders.empty:
        print("  ✓  No placeholder bsa00_s regions (mean ≤ €5)")
    else:
        print(f"  ✗  bsa00_s placeholder regions:")
        for _, r in imv_placeholders.iterrows():
            print(f"       {r['region']:<22} mean={r['mean_imv']:.2f} €/mo")

    if rmi_placeholders.empty:
        print("  ✓  No placeholder bsarg_s regions (mean ≤ €5)")
    else:
        print(f"  ✗  bsarg_s placeholder regions:")
        for _, r in rmi_placeholders.iterrows():
            print(f"       {r['region']:<22} mean={r['mean_rmi']:.2f} €/mo")

    imv_high = reg[reg["mean_imv"] > IMV_MAX_PLAUSIBLE]
    rmi_high = reg[reg["mean_rmi"] > RMI_MAX_PLAUSIBLE]
    if not imv_high.empty:
        print(f"  ~  bsa00_s unusually high (> €{IMV_MAX_PLAUSIBLE:.0f}/mo):")
        for _, r in imv_high.iterrows():
            print(f"       {r['region']:<22} mean={r['mean_imv']:.2f} €/mo")
    if not rmi_high.empty:
        print(f"  ~  bsarg_s unusually high (> €{RMI_MAX_PLAUSIBLE:.0f}/mo):")
        for _, r in rmi_high.iterrows():
            print(f"       {r['region']:<22} mean={r['mean_rmi']:.2f} €/mo")

    print(f"\n  Full regional table:")
    print(reg[[
        "region", "drgn2",
        "w_rec_imv", "mean_imv", "exp_imv_M",
        "w_rec_rmi", "mean_rmi", "exp_rmi_M",
        "exp_total_M",
    ]].to_string(index=False))

def print_cross_year_summary(yearly_data: dict[int, pd.DataFrame]) -> None:
    print("\n")
    print("=" * 60)
    print("  CROSS-YEAR SUMMARY — La Rioja & Aragón, both variables")
    print("=" * 60)

    for drgn2, name in [(23, "La Rioja"), (24, "Aragón")]:
        print(f"\n  {name}  (drgn2={drgn2})")
        print(f"  {'Year':<6} {'bsa00_s mean':>14} {'bsarg_s mean':>14} {'Total exp M€':>14}")
        print(f"  {'─'*6} {'─'*14} {'─'*14} {'─'*14}")
        for year in sorted(yearly_data):
            df  = yearly_data[year]
            reg = df[df["drgn2"] == drgn2].copy()
            reg["total_s"] = reg["bsa00_s"].fillna(0) + reg["bsarg_s"].fillna(0)

            def wmean_var(col):
                r = reg[reg[col] > 0]
                return (r[col] * r["dwt"]).sum() / r["dwt"].sum() if r["dwt"].sum() > 0 else float("nan")

            mn_imv = wmean_var("bsa00_s")
            mn_rmi = wmean_var("bsarg_s")
            exp_tot = (reg["total_s"] * reg["dwt"]).sum() * 12 / 1_000_000

            mn_imv_s = f"{mn_imv:.2f}" if not np.isnan(mn_imv) else "—"
            mn_rmi_s = f"{mn_rmi:.2f}" if not np.isnan(mn_rmi) else "—"
            print(f"  {year:<6} {mn_imv_s:>14} {mn_rmi_s:>14} {exp_tot:>14.3f}")

    print()
    print("  If bsa00_s mean ≈ €1 → national IMV bug persists in counterfactual.")
    print("  If bsarg_s mean ≈ €1 → regional RMI bug persists in counterfactual.")
    print("  If both are plausible → regions can be included via total_s = bsa00_s + bsarg_s.")
    print("  If either is €1 → total_s is unreliable → exclusion from DiD remains justified.")

def main() -> None:
    yearly_data: dict[int, pd.DataFrame] = {}

    for year, path in sorted(FILES.items()):
        print("\n")
        print("=" * 60)
        print(f"  YEAR {year}")
        print("=" * 60)

        try:
            df = load_euromod_output(path)
        except FileNotFoundError:
            print(f"  ERROR: file not found — {path}")
            continue

        required_cols = ["dwt", "bsa00_s", "bsarg_s", "drgn2"]
        missing_cols  = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"  ERROR: missing required columns: {missing_cols}")
            continue

        yearly_data[year] = df

        print_national_summary(year, df)
        print_implausible_checks(df)
        print_broken_region_diagnostics(year, df)
        print_benefit_distribution(year, df)

    if len(yearly_data) > 1:
        print_cross_year_summary(yearly_data)


if __name__ == "__main__":
    main()
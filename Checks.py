import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def load_euromod(path):
    df = pd.read_csv(path, sep='\t', low_memory=False, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(',', '.', regex=False).str.strip(),
            errors='coerce'
        )
    return df

REGION_NAMES = {
    11:'Galicia', 12:'Asturias', 13:'Cantabria',
    21:'País Vasco', 22:'Navarra', 30:'Madrid',
    41:'Castilla y León', 42:'Castilla-La Mancha', 43:'Extremadura',
    51:'Cataluña', 52:'C. Valenciana', 53:'Illes Balears',
    61:'Andalucía', 62:'Murcia', 70:'Canarias'
}

EXCLUDE      = {23, 24, 63, 64}
INCOMPATIBLE = {11, 53, 61}

RMI_FILES = {
    2017: "input_data/euromod_output/es_2017_std.txt",
    2018: "input_data/euromod_output/es_2018_std.txt",
    2019: "input_data/euromod_output/es_2019_std.txt",
}
IMV_FILES = {
    2017: "input_data/euromod_output/IMV_2022ruleson2017.txt",
    2018: "input_data/euromod_output/IMV_2022ruleson2018.txt",
    2019: "input_data/euromod_output/IMV_2022ruleson2019.txt",
}

# Administrative RMI generosity benchmark (avg monthly benefit from Informe)
# gasto_anual_por_titular / 12 — pooled average 2017-2019
ADMIN_RMI_MONTHLY = {
    11: (4064.81 + 3885.31 + 3925.85) / 3 / 12,   # Galicia
    12: (5361.19 + 5584.11 + 5501.72) / 3 / 12,   # Asturias
    13: (4902.52 + 5775.67 + 4260.35) / 3 / 12,   # Cantabria
    21: (6148.30 + 6062.42 + 6352.42) / 3 / 12,   # País Vasco
    22: (6161.69 + 6438.65 + 6561.06) / 3 / 12,   # Navarra
    30: (4752.32 + 4623.03 + 5421.78) / 3 / 12,   # Madrid
    41: (5013.16 + 4944.90 + 4903.90) / 3 / 12,   # Castilla y León
    42: (2129.21 + 2663.89 + 4296.13) / 3 / 12,   # Castilla-La Mancha
    43: (7668.46 + 8096.62 + 5935.93) / 3 / 12,   # Extremadura
    51: (7014.99 + 8417.56 + 8317.18) / 3 / 12,   # Cataluña
    52: (2707.61 + 2437.64 + 6279.73) / 3 / 12,   # C. Valenciana
    53: (1763.10 + 2196.11 + 2639.81) / 3 / 12,   # Illes Balears
    61: (2942.35 + 3003.66 + 4824.52) / 3 / 12,   # Andalucía
    62: (2609.24 + 2821.62 + 2906.34) / 3 / 12,   # Murcia
    70: (3923.00 + 3676.40 + 3663.09) / 3 / 12,   # Canarias
}

all_dims = []

for year in [2017, 2018, 2019]:
    rmi = load_euromod(RMI_FILES[year])
    imv = load_euromod(IMV_FILES[year])

    imv_gl = imv.copy()
    imv_gl.loc[imv_gl["drgn2"].isin(INCOMPATIBLE), "bsarg_s"] = 0.0

    # Check which counterfactual variables exist
    has_il_bsarg = "il_bsarg_global" in rmi.columns
    has_il_bsa   = "il_bsa_global"   in imv.columns

    print(f"Year {year}: il_bsarg_global in RMI = {has_il_bsarg} | "
          f"il_bsa_global in IMV = {has_il_bsa}")

    for drgn2 in sorted(rmi["drgn2"].dropna().unique()):
        if drgn2 in EXCLUDE:
            continue

        r   = rmi[rmi["drgn2"] == drgn2]
        i   = imv_gl[imv_gl["drgn2"] == drgn2]
        pop = r["dwt"].sum()

        # --- BCA-adjusted variables (current approach) ---
        rmi_rec_w  = r[r["bsarg_s"] > 0]["dwt"].sum()
        rmi_mean   = (
            (r[r["bsarg_s"]>0]["bsarg_s"] * r[r["bsarg_s"]>0]["dwt"]).sum()
            / max(rmi_rec_w, 1)
        )
        imv_rec_w  = i[i["bsa00_s"] > 0]["dwt"].sum()
        i["total"] = i["bsa00_s"] + i["bsarg_s"]
        imv_mean   = (
            (i[i["total"]>0]["total"] * i[i["total"]>0]["dwt"]).sum()
            / max(i[i["total"]>0]["dwt"].sum(), 1)
        )
        rmi_exp    = (r["bsarg_s"] * r["dwt"]).sum() * 12
        imv_exp    = (i["total"] * i["dwt"]).sum() * 12

        row = {
            "year": year, "drgn2": int(drgn2),
            "region": REGION_NAMES.get(int(drgn2), "?"),
            "pop": pop,
            "admin_rmi_monthly": ADMIN_RMI_MONTHLY.get(int(drgn2), np.nan),
            # BCA-adjusted
            "bca_rmi_mean":      round(rmi_mean, 2),
            "bca_imv_mean":      round(imv_mean, 2),
            "bca_delta_mean":    round(imv_mean - rmi_mean, 2),
            "bca_delta_rec_pc":  round((imv_rec_w - rmi_rec_w)/pop*100, 4),
            "bca_delta_exp_pc":  round((imv_exp - rmi_exp)/pop, 4),
        }

        # --- Pre-BCA counterfactual variables ---
        if has_il_bsarg and has_il_bsa:
            il_rmi_mean  = (
                (r[r["il_bsarg_global"]>0]["il_bsarg_global"] *
                 r[r["il_bsarg_global"]>0]["dwt"]).sum()
                / max(r[r["il_bsarg_global"]>0]["dwt"].sum(), 1)
            )
            il_rmi_rec_w = r[r["il_bsarg_global"] > 0]["dwt"].sum()
            il_rmi_exp   = (r["il_bsarg_global"] * r["dwt"]).sum() * 12

            # For IMV: zero il_bsarg_global for incompatible regions
            i2 = imv.copy()
            i2.loc[i2["drgn2"].isin(INCOMPATIBLE), "il_bsarg_global"] = 0.0
            i2 = i2[i2["drgn2"] == drgn2]
            i2["il_total"] = i2["il_bsa_global"] + i2["il_bsarg_global"]

            il_imv_mean  = (
                (i2[i2["il_total"]>0]["il_total"] *
                 i2[i2["il_total"]>0]["dwt"]).sum()
                / max(i2[i2["il_total"]>0]["dwt"].sum(), 1)
            )
            il_imv_rec_w = i2[i2["il_bsa_global"] > 0]["dwt"].sum()
            il_imv_exp   = (i2["il_total"] * i2["dwt"]).sum() * 12

            row.update({
                "il_rmi_mean":     round(il_rmi_mean, 2),
                "il_imv_mean":     round(il_imv_mean, 2),
                "il_delta_mean":   round(il_imv_mean - il_rmi_mean, 2),
                "il_delta_rec_pc": round((il_imv_rec_w - il_rmi_rec_w)/pop*100, 4),
                "il_delta_exp_pc": round((il_imv_exp - il_rmi_exp)/pop, 4),
            })

        all_dims.append(row)

df = pd.DataFrame(all_dims)
avg = df.groupby("drgn2").mean(numeric_only=True).reset_index()
avg["region"] = avg["drgn2"].map(REGION_NAMES)

print()
print("=" * 70)
print("INSTITUTIONAL CONSISTENCY CHECK")
print("Expected: negative rho (more generous pre-reform = less gain)")
print("=" * 70)

print("\n--- BCA-adjusted variables (current approach) ---")
rho1, p1 = spearmanr(avg["admin_rmi_monthly"], avg["bca_delta_exp_pc"])
rho2, p2 = spearmanr(avg["admin_rmi_monthly"], avg["bca_delta_rec_pc"])
print(f"Spearman(admin_rmi, bca_delta_exp_pc): rho={rho1:.3f} (p={p1:.4f})")
print(f"Spearman(admin_rmi, bca_delta_rec_pc): rho={rho2:.3f} (p={p2:.4f})")

if "il_delta_exp_pc" in avg.columns:
    print("\n--- Pre-BCA counterfactual variables (il_bsa_global) ---")
    rho3, p3 = spearmanr(avg["admin_rmi_monthly"], avg["il_delta_exp_pc"])
    rho4, p4 = spearmanr(avg["admin_rmi_monthly"], avg["il_delta_rec_pc"])
    print(f"Spearman(admin_rmi, il_delta_exp_pc): rho={rho3:.3f} (p={p3:.4f})")
    print(f"Spearman(admin_rmi, il_delta_rec_pc): rho={rho4:.3f} (p={p4:.4f})")
    print()
    print("Regional comparison BCA vs il_ variables:")
    cols = ["region", "admin_rmi_monthly",
            "bca_delta_exp_pc", "il_delta_exp_pc",
            "bca_delta_rec_pc", "il_delta_rec_pc"]
    print(avg[cols].sort_values("admin_rmi_monthly", ascending=False).to_string(index=False))
else:
    print("\nil_bsa_global / il_bsarg_global not found in files")
    print("Can only evaluate BCA approach")
    print()
    print("Regional BCA results vs admin RMI:")
    cols = ["region","admin_rmi_monthly","bca_delta_exp_pc","bca_delta_rec_pc"]
    print(avg[cols].sort_values("admin_rmi_monthly",ascending=False).to_string(index=False))
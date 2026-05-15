import pandas as pd

df = pd.read_csv(
    "input_data/euromod_output/IMV_2022ruleson2017.txt",
    sep="\t", nrows=1, dtype=str
)

# Print every column that contains bsa, bsarg, il_, or yn
candidates = [c for c in df.columns if any(
    x in c.lower() for x in ["bsa", "il_", "yn_", "_e", "_elig"]
)]
print("\n".join(sorted(candidates)))

for col in ["bsaec00_s", "il_bsawr", "bsa_s"]:
    if col in imv.columns:
        nonzero = (imv[col] > 0).sum()
        neg     = (imv[col] < 0).sum()
        mean_nz = imv.loc[imv[col] > 0, col].mean() if nonzero > 0 else 0
        print(f"{col:20s}: nonzero={nonzero:6d}, neg={neg:4d}, mean(>0)={mean_nz:.2f}")

# Also check alignment with eligibility flag
elig = imv["bsa00yn_a"] == 1
print(f"\nbsa00yn_a==1: {elig.sum()} persons")
for col in ["bsaec00_s", "bsa_s"]:
    if col in imv.columns:
        print(f"{col}>0 among eligible: {(imv.loc[elig, col] > 0).sum()}")
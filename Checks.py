import polars as pl
import pandas as pd
from src.constants import EXPOSURE_TERCILES

df = pl.read_parquet('output/analysis_dataset_with_gap.parquet').to_pandas()
keep = [2017,2018,2019,2021,2022,2023,2024,2025]
df = df[df['year'].isin(keep)].copy()

# Check which region is dropped as reference in get_dummies drop_first=True
regions_sorted = sorted(df['drgn2'].unique().tolist())
ref_region = regions_sorted[0]

print(f"Reference region (dropped): {ref_region}")
print(f"Low tercile:    {EXPOSURE_TERCILES['low']}")
print(f"Medium tercile: {EXPOSURE_TERCILES['medium']}")
print(f"High tercile:   {EXPOSURE_TERCILES['high']}")
print(f"Is ref in low? {ref_region in EXPOSURE_TERCILES['low']}")
print(f"Is ref in medium? {ref_region in EXPOSURE_TERCILES['medium']}")
print(f"Is ref in high? {ref_region in EXPOSURE_TERCILES['high']}")

# Check mean of each control by tercile
low_r    = EXPOSURE_TERCILES['low']
med_r    = EXPOSURE_TERCILES['medium']
high_r   = EXPOSURE_TERCILES['high']

for ctrl in ['head_age_under35','head_age_55_64','head_age_65plus',
             'head_high_education','head_unemployed','single_parent_hh','hh_size']:
    low_m  = df[df['drgn2'].isin(low_r)][ctrl].mean()
    med_m  = df[df['drgn2'].isin(med_r)][ctrl].mean()
    high_m = df[df['drgn2'].isin(high_r)][ctrl].mean()
    print(f"{ctrl:25s}  low={low_m:.3f}  med={med_m:.3f}  high={high_m:.3f}")
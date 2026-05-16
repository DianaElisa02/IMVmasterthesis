import polars as pl
df = pl.read_parquet('output/analysis_dataset.parquet')
print(df['income_net_annual'].null_count(), '/', len(df))
print(df['income_net_annual'].describe())
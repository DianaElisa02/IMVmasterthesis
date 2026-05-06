import pandas as pd
df = pd.read_csv('/workspaces/IMVmasterthesis/input_data/euromod_output/es_2017_std.txt', sep='\t', decimal=',')

for year, path in FILES.items():
    df = load_euromod_output(path)
    over65 = df[(df['bsarg_s'] > 0) & (df['dag'] > 65)]
    print(f"\nYear {year} — over-65 recipients by region:")
    print(over65.groupby('drgn2')['dwt'].sum().round(0))
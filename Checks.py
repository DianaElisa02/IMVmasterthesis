import pandas as pd
df = pd.read_csv('/workspaces/IMVmasterthesis/input_data/euromod_output/es_2017_std.txt', sep='\t', decimal=',')

print("dwt dtype:", df["dwt"].dtype)
print("dwt sample values:", df["dwt"].head(10).tolist())
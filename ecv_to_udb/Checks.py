import pandas as pd
df = pd.read_csv("output/ES_2018_a1.txt", sep="\t", low_memory=False, nrows=2)
print([c for c in df.columns if "drgn" in c.lower()])
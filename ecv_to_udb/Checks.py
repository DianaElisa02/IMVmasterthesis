import pandas as pd
th = pd.read_stata("input_data/ECV_Th_2017.dta", convert_categoricals=False)
print([c for c in th.columns if "HX" in c.upper()])

th = pd.read_stata("input_data/ECV_Th_2017.dta", convert_categoricals=False)
print(th["HX040"].head(10).tolist())
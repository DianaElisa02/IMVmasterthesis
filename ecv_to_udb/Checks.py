import pandas as pd
tp = pd.read_stata("input_data/ECV_Tp_2017.dta", convert_categoricals=False)
print([c for c in tp.columns if "PL03" in c.upper() or "PE04" in c.upper()])
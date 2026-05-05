import pandas as pd
td = pd.read_stata("input_data/ECV_Td_2017.dta", convert_categoricals=False)
print(td["DB030"].head(10).tolist())
print("dtype:", td["DB030"].dtype)
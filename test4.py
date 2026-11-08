import pandas as pd
print(pd.read_csv("data/processed/dpe_clean.csv", nrows=1).columns.tolist()[-5:])
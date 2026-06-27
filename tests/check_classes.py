import pandas as pd
df = pd.read_csv('notebooks/data/raw/tabular/dataset.csv')
classes = sorted(df.iloc[:, 0].unique())
for c in classes:
    if 'hiv' in c.lower() or 'aid' in c.lower():
        print("MATCH:", c)

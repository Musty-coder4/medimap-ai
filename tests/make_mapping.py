import pandas as pd
import json

df = pd.read_csv('notebooks/data/raw/tabular/dataset.csv')
disease_symptoms = {}

for index, row in df.iterrows():
    disease = str(row.iloc[0]).strip()
    symptoms = [str(x).strip() for x in row.iloc[1:].dropna().values if str(x).strip()]
    if disease not in disease_symptoms:
        disease_symptoms[disease] = set(symptoms)
    else:
        disease_symptoms[disease].update(symptoms)

# convert sets to lists
for k in disease_symptoms:
    disease_symptoms[k] = list(disease_symptoms[k])

with open('models/disease_symptoms.json', 'w') as f:
    json.dump(disease_symptoms, f, indent=4)
print("Created disease_symptoms.json")

import pandas as pd
import numpy as np

df = pd.read_csv('notebooks/data/raw/tabular/dataset.csv')
disease_symptoms = {}

for index, row in df.iterrows():
    disease = row.iloc[0]
    # Drop NaNs and strip whitespace
    symptoms = [str(x).strip() for x in row.iloc[1:].dropna().values if str(x).strip()]
    if disease not in disease_symptoms:
        disease_symptoms[disease] = set(symptoms)
    else:
        disease_symptoms[disease].update(symptoms)

# Convert sets to sorted lists and get lengths
results = []
for disease, symptoms in disease_symptoms.items():
    results.append({
        'Disease': disease,
        'Symptom_Count': len(symptoms),
        'Symptoms': sorted(list(symptoms))
    })

# Sort by symptom count
results = sorted(results, key=lambda x: x['Symptom_Count'])

print("DISEASES WITH THE FEWEST SYMPTOMS:")
print("-" * 50)
for r in results[:10]:
    print(f"{r['Disease']} ({r['Symptom_Count']} symptoms):")
    print(f"  {', '.join(r['Symptoms'])}")
    print()

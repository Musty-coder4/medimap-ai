import json

symptoms = json.load(open('data/processed/symptom_columns.json'))
with open('utils/symptom_extractor.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to replace the OFFLINE_SYNONYM_MAP. 
import re
new_map = "OFFLINE_SYNONYM_MAP = {\n"
for s in symptoms:
    # basic mapping: exact name
    new_map += f'    "{s.replace("_", " ")}": "{s}",\n'
    # some basic synonyms
    if "pain" in s:
        new_map += f'    "{s.replace("pain", "ache").replace("_", " ")}": "{s}",\n'
        new_map += f'    "{s.replace("pain", "hurting").replace("_", " ")}": "{s}",\n'
new_map += "}\n\n"

# regex replace
content = re.sub(r'OFFLINE_SYNONYM_MAP = \{.*?(?=\n[A-Za-z_])', new_map, content, flags=re.DOTALL)

with open('utils/symptom_extractor.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated symptom extractor!")

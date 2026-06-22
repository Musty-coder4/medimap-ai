import json
from utils.symptom_extractor import extract_symptoms_from_text

# Load actual symptoms list used by the app
try:
    with open("data/processed/symptom_columns.json", "r") as f:
        all_symptoms = json.load(f)
except Exception as e:
    print(f"Could not load symptoms: {e}")
    all_symptoms = [
        "high_fever", "mild_fever", "cough", "breathlessness", "runny_nose",
        "congestion", "throat_irritation", "chest_pain", "headache", "fatigue",
        "muscle_pain", "nausea", "vomiting", "diarrhoea", "stomach_pain",
        "dizziness", "shivering", "chills", "loss_of_appetite", "skin_rash",
        "itching", "joint_pain", "back_pain", "sweating", "yellowish_skin",
        "redness_of_eyes", "spinning_movements", "phlegm", "dehydration",
        "weight_loss", "continuous_sneezing", "loss_of_smell",
    ]

test = "I have been running temperature since yesterday, my head is paining me and my body dey weak. I also cannot eat anything."

print(f"Testing with {len(all_symptoms)} symptoms")
matched, explanations, engine = extract_symptoms_from_text(test, all_symptoms)
print(f"Engine : {engine.upper()}")
print(f"Matched: {matched}")
print(f"Count  : {len(matched)}")

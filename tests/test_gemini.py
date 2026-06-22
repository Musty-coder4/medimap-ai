import os

from utils.symptom_extractor import extract_symptoms_from_text

SYMPTOMS = [
    "high_fever", "mild_fever", "cough", "breathlessness", "runny_nose",
    "congestion", "throat_irritation", "chest_pain", "headache", "fatigue",
    "muscle_pain", "nausea", "vomiting", "diarrhoea", "stomach_pain",
    "dizziness", "shivering", "chills", "loss_of_appetite", "skin_rash",
    "itching", "joint_pain", "back_pain", "sweating", "yellowish_skin",
]

test = "I have been running temperature since yesterday, my head is paining me and my body dey weak. I also cannot eat anything."

matched, explanations, engine = extract_symptoms_from_text(test, SYMPTOMS)
print(f"Engine : {engine.upper()}")
print(f"Matched: {matched}")
print(f"Count  : {len(matched)}")

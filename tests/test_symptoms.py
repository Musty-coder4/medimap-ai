"""
Live test of the enhanced symptom extractor — v2.0
"""
from utils.symptom_extractor import extract_symptoms_from_text

# Simulated symptom list (subset of real model columns)
SYMPTOMS = [
    "high_fever", "mild_fever", "cough", "breathlessness", "runny_nose",
    "congestion", "throat_irritation", "chest_pain", "headache", "fatigue",
    "muscle_pain", "nausea", "vomiting", "diarrhoea", "stomach_pain",
    "dizziness", "shivering", "chills", "loss_of_appetite", "skin_rash",
    "itching", "joint_pain", "back_pain", "sweating", "yellowish_skin",
    "redness_of_eyes", "spinning_movements", "phlegm", "dehydration",
    "weight_loss", "continuous_sneezing", "loss_of_smell",
]

tests = [
    # Nigerian English
    ("I have been running temperature since yesterday and my head is paining me", "Should get: high_fever, headache"),
    # Pidgin
    ("My body dey weak and e dey hot", "Should get: fatigue, high_fever"),
    # Negation (should NOT match fever)
    ("I don't have a fever but I feel very weak and dizzy", "Should get: fatigue, dizziness — NOT fever"),
    # Vague / indirect
    ("My chest feels like there's a brick on it and I can't breathe properly", "Should get: chest_pain, breathlessness"),
    # Multi-symptom narrative
    ("I woke up dizzy, my stomach is making noise and I cannot eat anything all day", "Should get: dizziness, stomach_pain, loss_of_appetite"),
    # Classic simple input (regression test)
    ("I have a high fever, cough and sore throat", "Should get: high_fever, cough, throat_irritation"),
]

print("=" * 72)
for text, note in tests:
    matched, explanations, engine = extract_symptoms_from_text(text, SYMPTOMS)
    print(f"\nINPUT: {text[:65]}...")
    print(f"  NOTE:   {note}")
    print(f"  ENGINE: {engine.upper()}")
    print(f"  FOUND:  {', '.join(matched) if matched else '(none)'}")
print("=" * 72)

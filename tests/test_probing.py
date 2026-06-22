import torch
import numpy as np
from models.hybrid_fusion import MediMapHybridModel
from models.inference_engine import MediMapInferenceEngine

engine = MediMapInferenceEngine(
    model_checkpoint="models/saved/medimap_best.pth",
    label_encoder_path="models/saved/label_encoder.pkl",
    symptom_columns_path="data/processed/symptom_columns.json",
)

def get_clarifying_symptoms(engine, current_symptoms, top_n=5):
    # Baseline
    baseline_res = engine.predict_from_inputs(current_symptoms)
    base_probs = baseline_res['probabilities'] # dict mapping class to prob
    top_class = baseline_res['predicted_class']
    base_conf = baseline_res['confidence']
    
    print(f"Baseline: {top_class} ({base_conf:.2f})")
    
    unselected = [s for s in engine.symptom_columns if s not in current_symptoms]
    
    scores = []
    
    for sym in unselected:
        test_syms = current_symptoms + [sym]
        res = engine.predict_from_inputs(test_syms)
        
        # Method 1: Does this symptom increase the confidence of the CURRENT top class?
        top_idx = baseline_res['class_index']
        shift = res['probabilities'][top_idx] - base_probs[top_idx]
        scores.append((sym, shift))
        
    # Sort by highest shift
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

test_syms = ["headache", "muscle_pain", "chills"]
print("Testing with:", test_syms)
questions = get_clarifying_symptoms(engine, test_syms)
print("\nTop 5 clarifying symptoms to ask about:")
for sym, score in questions:
    print(f"- {sym}: shift score = {score:.4f}")

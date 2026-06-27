import os
import torch
import json
import numpy as np
import joblib

from utils.symptom_extractor import extract_symptoms_from_text
from models.hybrid_fusion import MediMapHybridModel

text = "my body is paining me and i am feeling headache and aslo i feel cold and my body is shivering"

# 1. Extract symptoms
symptoms = extract_symptoms_from_text(text)
print(f"Extracted symptoms: {symptoms}")

# 2. Load model
with open('data/processed/symptom_columns.json') as f:
    symptom_cols = json.load(f)

label_encoder = joblib.load('models/saved/label_encoder.pkl')
label_names = label_encoder.classes_.tolist()

model = MediMapHybridModel(
    tabular_input_dim=len(symptom_cols),
    num_classes=len(label_names),
    vision_backbone='resnet50'
)
model.eval()

ckpt = torch.load('models/saved/medimap_best.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)

# 3. Create tabular vector
vec = np.zeros(len(symptom_cols), dtype=np.float32)
for s in symptoms:
    if s in symptom_cols:
        vec[symptom_cols.index(s)] = 1.0

tab_tensor = torch.tensor(vec).unsqueeze(0)
img_tensor = torch.zeros((1, 3, 224, 224))

with torch.no_grad():
    logits, _ = model(img_tensor, tab_tensor)
    probs = torch.softmax(logits, dim=1).numpy()[0]
    
idx = np.argmax(probs)
print(f"Predicted: {label_names[idx]} ({probs[idx]*100:.2f}%)")

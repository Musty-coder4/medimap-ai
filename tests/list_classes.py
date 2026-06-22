from models.inference_engine import MediMapInferenceEngine

engine = MediMapInferenceEngine(
    model_checkpoint="models/saved/medimap_best.pth",
    label_encoder_path="models/saved/label_encoder.pkl",
    symptom_columns_path="data/processed/symptom_columns.json",
)

print(engine.label_names)

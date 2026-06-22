# MediMap AI

> **Hybrid Multi-Modal Healthcare Expert System & GIS Specialist Recommender**

A production-grade AI system that fuses symptom checklists (tabular) with medical imagery (X-rays / skin scans) using a **Dual-Stream Late-Fusion Deep Learning model**, then maps predictions to localised specialist clinics via real-time **Geospatial APIs**.

---

## 🏗️ Project Architecture

```
medimap-ai/
├── app/
│   └── main.py                  # Streamlit dashboard UI
├── models/
│   ├── hybrid_fusion.py         # ★ Dual-Stream Late-Fusion model (PyTorch)
│   ├── train.py                 # Training pipeline + AMP + checkpointing
│   ├── evaluate.py              # Metrics: accuracy, F1, AUROC
│   ├── inference_engine.py      # Production inference wrapper
│   └── saved/                   # Serialised .pth checkpoints
├── utils/
│   ├── geo_recommender.py       # ★ GIS pipeline (Overpass/OSM + Folium)
│   └── data_preprocessor.py    # Tabular + image preprocessing helpers
├── data/
│   ├── raw/tabular/             # Kaggle Disease Symptom CSV
│   ├── raw/images/xray/         # Chest X-ray images
│   └── raw/images/skin/         # Skin lesion images
├── tests/
│   └── test_medimap.py          # Pytest unit test suite
├── requirements.txt
├── setup_project.py
└── .env.example
```

---

## 🚀 Quick Start

### 1 · Install dependencies
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt
```

### 2 · Scaffold project directories
```powershell
python setup_project.py
```

### 3 · Add your dataset
Download the **Kaggle Disease Symptom & Patient Profile** dataset and place the CSV at:
```
data/raw/tabular/dataset.csv
```
Dataset URL: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

### 4 · Train the model
```powershell
python models/train.py `
  --tabular_csv data/raw/tabular/dataset.csv `
  --epochs 30 `
  --batch_size 32 `
  --backbone mobilenet_v2 `
  --unfreeze_epoch 5
```

### 5 · Launch the Streamlit app
```powershell
streamlit run app/main.py
```

---

## 🧠 Model Architecture

```
                     ┌─────────────────────────────────┐
 Symptom Vector ───► │  Stream A: TabularMLP            │ ──► 128-d embedding
 (one-hot, 131d)     │  (Linear → BN → ReLU × 3)       │
                     └─────────────────────────────────┘
                                                          ╲
                                                           ╲
                                                            ► Concat (256-d) ──► FusionHead ──► Softmax
                                                           ╱                      (Linear × 3)    (41 classes)
                     ┌─────────────────────────────────┐ ╱
 Medical Image ────► │  Stream B: VisionCNN             │ ──► 128-d embedding
 (224×224, RGB)      │  (MobileNetV2 + Projection Head) │
                     └─────────────────────────────────┘
```

| Component | Details |
|-----------|---------|
| Tabular stream | MLP (512→256→128) with BatchNorm + Dropout |
| Vision stream | MobileNetV2 / ResNet50 (pretrained ImageNet) |
| Fusion | Late-fusion concat → 256-d → 128-d → `num_classes` |
| Training | AdamW + CosineAnnealingLR + AMP mixed precision |
| Regularisation | Label smoothing (0.1), Dropout (0.3), Grad clipping |

---

## 🗺️ GIS Pipeline

```
Disease Prediction
      │
      ▼
disease_to_specialty()        # e.g. "Pneumonia" → "Pulmonologist"
      │
      ▼
geocode_address()             # Nominatim → (lat, lon)
      │
      ▼
search_specialists()          # Overpass API / OSM → clinic list
      │
      ▼
build_folium_map()            # Interactive map with numbered markers
```

---

## ⚙️ Configuration

Copy `.env.example` → `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEO_SEARCH_RADIUS_M` | `10000` | Clinic search radius (metres) |
| `GEO_MAX_RESULTS` | `10` | Max clinics returned |
| `OVERPASS_API_URL` | OSM public | Overpass API endpoint |
| `FORCE_CPU` | `0` | Set `1` to disable CUDA |

---

## 🧪 Running Tests

```powershell
pytest tests/ -v --tb=short --cov=models --cov=utils
```

---

## ⚠️ Disclaimer

MediMap AI is a **research and educational prototype**.
It is **not a certified medical device** and must **not** be used for clinical diagnosis.
Always consult a qualified medical professional.

---

## 📋 License

MIT License — see `LICENSE` for details.

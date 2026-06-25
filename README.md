# 🩺 MediMap AI: Late-Fusion Multi-Modal Healthcare System & GIS Recommender

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

MediMap AI is an advanced, multi-modal healthcare expert system built as a university Final Year Project. It merges **Deep Learning**, **Generative AI**, and **Geographic Information Systems (GIS)** to provide highly accurate disease predictions, personalized medical insights, and live physical hospital recommendations based on the user's location.

---

## ✨ Key Features

1. **Late-Fusion Multi-Modal Architecture:** 
   - Uses a **Multi-Layer Perceptron (MLP)** for processing 131 clinical symptoms.
   - Uses a **ResNet-50 Convolutional Neural Network (CNN)** for processing medical imaging (Chest X-Rays and Dermatoscopic Skin Lesions).
   - Fuses both modalities at the final layers to predict up to 41 distinct diseases with exceptional accuracy.
   
2. **Generative AI Health Insights:** 
   - Integrated with the **Google Gemini API** (`gemini-2.5-flash-lite`).
   - Extracts medical keywords natively from free-text user descriptions.
   - Generates personalized, dynamically typed health insights, precautions, and actionable advice post-prediction.

3. **GIS Specialist Recommender:** 
   - Utilizes the **Overpass API** (OpenStreetMap) to dynamically locate nearby healthcare facilities.
   - Automatically maps predictions to specific medical specialists (e.g., *Dermatologist* for Acne, *Pulmonologist* for Pneumonia).
   - Features an **Intelligent Fallback System** that defaults to General Practitioners and General Hospitals if specialized clinics are unavailable in the user's immediate radius.

4. **Web Deployment:** 
   - Fully deployed via **Streamlit Community Cloud**.
   - Handles massive 280MB+ PyTorch model checkpoints efficiently using **Git Large File Storage (LFS)**.
   - Automatically detects CPU vs CUDA availability to optimize inference speed.

---

## 🛠️ Technology Stack

| Domain | Technologies |
|---|---|
| **Deep Learning** | PyTorch, Torchvision, Scikit-learn |
| **Model Architectures** | ResNet-50, Multi-Layer Perceptron (MLP) |
| **Generative AI** | Google Gemini API |
| **GIS & Mapping** | Folium, Geopy, Overpass API, Nominatim |
| **Web Frontend** | Streamlit, Streamlit-Folium |

---

## 🚀 Running the Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Musty-coder4/medimap-ai.git
cd medimap-ai
```

### 2. Install Dependencies
Ensure you are using Python 3.12.
```bash
pip install -r requirements.txt
```

### 3. Add Environment Variables
Create a `.env` file in the root directory and add your free Gemini API key:
```env
GEMINI_API_KEY="your-api-key-here"
```

### 4. Run the Streamlit Application
```bash
streamlit run app/main.py
```

---

## 📊 Dataset & Training

The system was trained on a robust, multi-modal dataset consisting of:
- **Tabular Data:** 41 distinct diseases mapped against 131 unique symptoms (achieved 100% validation accuracy).
- **Vision Data:** Over 9,000 real medical images sourced from Kaggle.
  - *Chest X-Rays:* Pneumonia and Normal (Common Cold proxy).
  - *Skin Lesions (HAM10000 / ISIC):* Acne, Impetigo, Psoriasis, Fungal Infections.

---

## ⚠️ Medical Disclaimer

**MediMap AI is an academic research project.** The predictions, AI-generated insights, and clinic recommendations provided by this application are for informational and educational purposes only. They do not constitute professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

---

*Designed and engineered for Final Year Project Submission — 2026.*

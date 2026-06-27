"""
MediMap AI — Main Application (Page 1: Input)
===============================================
Handles symptom input, image upload, location selection,
model inference and GIS lookup, then navigates to the Analysis page.

Author : MediMap AI Engineering Team
Version: 3.0.0
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.hybrid_fusion import DEVICE, MediMapHybridModel, get_image_transforms
from utils.geo_recommender import recommend_specialists, DISEASE_SPECIALTY_MAP
from utils.symptom_extractor import extract_symptoms_from_text, GEMINI_API_KEY as _GEMINI_KEY

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediMap AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Fallback disease list (used in demo mode) ─────────────────────────────────
ALL_DISEASES: list[str] = sorted(DISEASE_SPECIALTY_MAP.keys())


# =============================================================================
# STYLE INJECTION  (light mode — fixed, no toggle)
# =============================================================================

def inject_style() -> None:
    """Inject a clean, fixed light-mode stylesheet."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── App shell ──────────────────────────────────────────────────── */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="block-container"],
    .main .block-container {
        background-color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Sidebar ────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    /* ── Text ───────────────────────────────────────────────────────── */
    html, body,
    .stApp, .stMarkdown, .stText,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    .stMarkdown p, .stMarkdown span, .stMarkdown li,
    p, span, label, li, td, th,
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }

    /* ── Widget labels ──────────────────────────────────────────────── */
    .stSelectbox label, .stMultiSelect label,
    .stSlider label, .stRadio label,
    .stCheckbox label, .stToggle label,
    .stTextInput label, .stNumberInput label,
    .stFileUploader label, .stTextArea label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
    }

    /* ── Captions ───────────────────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"],
    .stCaption p, small {
        color: #64748b !important;
    }

    /* ── Metric cards ───────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,.08) !important;
    }
    [data-testid="stMetricValue"]         { color: #2563eb !important; }
    [data-testid="stMetricLabel"] p       { color: #64748b !important; }

    /* ── Text inputs ────────────────────────────────────────────────── */
    input, textarea, select {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-color: #e2e8f0 !important;
    }
    input::placeholder, textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    [data-baseweb="base-input"],
    [data-baseweb="base-input"] > div,
    [data-baseweb="base-input"] input,
    [data-baseweb="textarea"] textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-color: #e2e8f0 !important;
    }

    /* ── Select / multiselect ───────────────────────────────────────── */
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div,
    [data-baseweb="select"] > div > div > div,
    [data-baseweb="select"] * {
        background-color: #ffffff !important;
        border-color: #e2e8f0 !important;
    }
    [data-baseweb="select"]:hover > div,
    [data-baseweb="select"]:focus-within > div {
        border-color: #2563eb !important;
        box-shadow: none !important;
    }
    [data-baseweb="select"] [data-baseweb="placeholder"],
    [data-baseweb="select"] span[class*="placeholder"],
    [data-baseweb="placeholder"],
    .stMultiSelect [data-baseweb="select"] span {
        color: #64748b !important;
        opacity: 1 !important;
    }
    [data-baseweb="select"] input,
    [data-baseweb="select"] input[type="text"],
    .stMultiSelect input, .stSelectbox input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        caret-color: #0f172a !important;
    }
    [data-baseweb="select"] input::placeholder,
    .stMultiSelect input::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }
    [data-baseweb="select"] [data-baseweb="single-value"],
    [data-baseweb="select"] span[class*="singleValue"] {
        color: #0f172a !important;
        background-color: transparent !important;
    }

    /* Tags (selected pills) */
    [data-baseweb="tag"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    [data-baseweb="tag"] span,
    [data-baseweb="tag"] svg { color: #ffffff !important; }

    /* Dropdown portal */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"],
    [role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    [role="option"],
    [data-baseweb="menu-item"],
    li[role="option"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    [role="option"]:hover,
    [data-baseweb="menu-item"]:hover {
        background-color: #f1f5f9 !important;
    }

    /* ── File uploader ──────────────────────────────────────────────── */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #e2e8f0 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #2563eb !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #2563eb, #6366f1) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] p {
        color: #64748b !important;
    }

    /* ── Radio / Checkbox ───────────────────────────────────────────── */
    [data-testid="stRadio"] label span,
    [data-testid="stCheckbox"] label span,
    .stRadio > div > label > div > p,
    .stCheckbox > label > div > p {
        color: #0f172a !important;
    }

    /* ── Alerts ─────────────────────────────────────────────────────── */
    [data-testid="stAlert"] > div, .stAlert {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] code { color: #0f172a !important; }

    /* ── Buttons ────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #6366f1) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: opacity .2s, transform .15s !important;
    }
    .stButton > button:hover {
        opacity: .92 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Dataframe ──────────────────────────────────────────────────── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    [data-testid="stDataFrameResizable"] { background-color: #ffffff !important; }
    .stDataFrame table, .stDataFrame th, .stDataFrame td {
        color: #0f172a !important;
        border-color: #e2e8f0 !important;
    }

    hr { border-color: #e2e8f0 !important; }

    /* ── Custom cards ───────────────────────────────────────────────── */
    .mm-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,.08);
        color: #0f172a !important;
    }
    .mm-card h3, .mm-card p, .mm-card span { color: #0f172a !important; }

    /* ── Badges ─────────────────────────────────────────────────────── */
    .badge {
        border-radius: 999px; padding: .25rem .75rem;
        font-size: .78rem; font-weight: 600; display: inline-block;
    }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-warn  { background: #fef3c7; color: #92400e; }
    .badge-blue  { background: #dbeafe; color: #1e40af; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# HERO BANNER
# =============================================================================

def render_hero() -> None:
    """Render the light-mode hero banner."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 60%, #ecfdf5 100%);
        border-radius: 20px; padding: 2.5rem 2rem; margin-bottom: 2rem;
        text-align: center; border: 1px solid #bae6fd;
        box-shadow: 0 4px 24px rgba(37,99,235,.08);">
        <div style="font-size:2.8rem;margin-bottom:.4rem">&#x2695;&#xFE0F;</div>
        <h1 style="color:#2563eb;font-size:2.4rem;font-weight:800;
                   margin:0 0 .5rem;letter-spacing:-.5px">
            MediMap AI
        </h1>
        <p style="color:#64748b;font-size:1.05rem;margin:0 0 1.2rem">
            Hybrid Multi-Modal Healthcare Expert System &amp; GIS Specialist Recommender
        </p>
        <div style="display:flex;justify-content:center;gap:.6rem;flex-wrap:wrap">
            <span style="background:rgba(37,99,235,.12);color:#1d4ed8;
                border:1px solid rgba(37,99,235,.25);border-radius:999px;
                padding:.25rem .9rem;font-size:.8rem;font-weight:600">
                &#x1F9E0; Deep Learning
            </span>
            <span style="background:rgba(5,150,105,.12);color:#065f46;
                border:1px solid rgba(5,150,105,.25);border-radius:999px;
                padding:.25rem .9rem;font-size:.8rem;font-weight:600">
                &#x1F5FA;&#xFE0F; GIS Mapping
            </span>
            <span style="background:rgba(245,158,11,.12);color:#92400e;
                border:1px solid rgba(245,158,11,.25);border-radius:999px;
                padding:.25rem .9rem;font-size:.8rem;font-weight:600">
                &#x26A1; CUDA Accelerated
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> dict:
    """Render configuration sidebar. Returns config dict."""
    with st.sidebar:
        st.markdown(
            "<h2 style='margin:0 0 1rem;font-size:1.2rem'>⚙️ Configuration</h2>",
            unsafe_allow_html=True,
        )

        # ── Compute ───────────────────────────────────────────────────────
        st.markdown("#### 🖥️ Compute")
        use_gpu = st.toggle(
            "Use GPU (CUDA)",
            value=torch.cuda.is_available(),
            disabled=not torch.cuda.is_available(),
        )
        if torch.cuda.is_available():
            st.caption(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.caption("⚠️ CUDA not available — using CPU")

        # ── Model ─────────────────────────────────────────────────────────
        st.markdown("#### 🤖 Model")
        backbone = "resnet50"
        st.caption("Vision Backbone: ResNet-50")
        show_chart = st.checkbox("Show Probability Chart", value=True)

        # ── GIS Settings ──────────────────────────────────────────────────
        st.markdown("#### 🗺️ GIS Settings")
        radius_km   = st.slider("Search Radius (km)", 1, 50, 10, step=1)
        max_clinics = st.slider("Max Clinics Shown",  1, 20,  8, step=1)
        map_theme   = st.radio(
            "Map Theme",
            ["CartoDB Positron", "OpenStreetMap"],
            index=0,
            horizontal=False,
        )

        # ── Gemini AI Status ───────────────────────────────────────────────
        st.markdown("#### 🤖 Gemini AI Status")
        try:
            from utils.symptom_extractor import GEMINI_API_KEY, _GEMINI_MODEL
            if GEMINI_API_KEY:
                st.caption(f"✅ Key loaded — Model: `{_GEMINI_MODEL}`")
                # Quick live ping to check the key actually works
                try:
                    from google import genai
                    _c = genai.Client(api_key=GEMINI_API_KEY)
                    _c.models.generate_content(
                        model=_GEMINI_MODEL,
                        contents="Reply with the single word: OK"
                    )
                    st.caption("🟢 Gemini API reachable")
                except Exception as _ge:
                    err_msg = str(_ge).split('.')[0] + '.' if '.' in str(_ge) else str(_ge)
                    st.caption(f"🔴 Gemini API error: {err_msg}")
            else:
                st.caption("🔴 No API key found in secrets")
        except Exception as _ie:
            st.caption(f"🔴 Import error: {_ie}")

        st.divider()
        st.caption("MediMap AI v3.0")

    return {
        "use_gpu"    : use_gpu,
        "backbone"   : backbone,
        "show_chart" : show_chart,
        "radius_km"  : radius_km,
        "max_clinics": max_clinics,
        "map_theme"  : map_theme,
    }


# =============================================================================
# RESOURCE LOADERS  (cached)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_symptom_columns() -> list[str]:
    path = ROOT / "data" / "processed" / "symptom_columns.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    logger.warning("symptom_columns.json not found — using fallback list.")
    return ["fatigue", "fever", "cough", "headache", "nausea",
            "chest_pain", "shortness_of_breath", "dizziness"]


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    path = ROOT / "models" / "saved" / "label_encoder.pkl"
    if path.exists():
        return joblib.load(path)
    logger.warning("label_encoder.pkl not found — demo mode.")
    return None


@st.cache_resource(show_spinner=False)
def load_model(
    n_symptoms: int,
    n_classes: int,
    backbone: str = "mobilenet_v2",
) -> Optional[MediMapHybridModel]:
    ckpt_path = ROOT / "models" / "saved" / "medimap_best.pth"
    if not ckpt_path.exists():
        logger.warning("No checkpoint found at %s — demo mode.", ckpt_path)
        return None
    try:
        model = MediMapHybridModel(
            tabular_input_dim=n_symptoms,
            num_classes=n_classes,
            vision_backbone=backbone,
        ).to(DEVICE)
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        logger.info("Model loaded from %s", ckpt_path)
        return model
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        import streamlit as st
        st.sidebar.error(f"Debug - Model Load Error: {exc}")
        return None


# =============================================================================
# INFERENCE HELPERS
# =============================================================================

def build_symptom_vector(selected: list[str], all_symptoms: list[str]) -> np.ndarray:
    vec = np.zeros(len(all_symptoms), dtype=np.float32)
    sym_set = set(selected)
    for i, sym in enumerate(all_symptoms):
        if sym in sym_set:
            vec[i] = 1.0
    return vec


def preprocess_image(file_obj) -> torch.Tensor:
    transform = get_image_transforms(split="val")
    img = Image.open(file_obj).convert("RGB")
    return transform(img)


DISEASE_PRIORS = {
    "Malaria": 1.5,
    "Common Cold": 1.5,
    "Typhoid": 1.5,
    "Dengue": 1.5,
    "Fungal infection": 1.5,
    "Allergy": 1.5,
    "Gastroenteritis": 1.5,
    "Acne": 1.5,
    "Paralysis (brain hemorrhage)": 1.5,
    "AIDS": 1.5,
    "Heart attack": 1.5,
}

def run_inference(
    model: Optional[MediMapHybridModel],
    sym_vec: np.ndarray,
    img_tensor: torch.Tensor,
    label_names: list[str],
) -> dict:
    if model is None:
        rng   = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(len(label_names)) * 0.3)
        idx   = int(np.argmax(probs))
        return {
            "predicted_class": label_names[idx],
            "confidence"     : float(probs[idx] * 100),
            "probabilities"  : probs.tolist(),
            "demo"           : True,
        }
    sym_t = torch.from_numpy(sym_vec).unsqueeze(0).to(DEVICE)
    img_t = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, probs_t = model(sym_t, img_t)
    probs = probs_t.squeeze(0).cpu().numpy()
    
    # Apply Bayesian Priors (Prevalence Weighting)
    weights = np.ones(len(label_names), dtype=np.float32)
    for i, name in enumerate(label_names):
        if name in DISEASE_PRIORS:
            weights[i] = DISEASE_PRIORS[name]
            
    calibrated_probs = probs * weights
    # Re-normalize so probabilities sum to 1.0
    calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
    
    idx   = int(np.argmax(calibrated_probs))
    return {
        "predicted_class": label_names[idx],
        "confidence"     : float(calibrated_probs[idx] * 100),
        "probabilities"  : calibrated_probs.tolist(),
        "demo"           : False,
    }


def get_clarifying_questions(
    model: Optional[MediMapHybridModel],
    current_symptoms: list[str],
    all_symptoms: list[str],
    label_names: list[str],
    base_res: dict,
    img_tensor: torch.Tensor,
    top_n: int = 10,
    target_rank: int = 0
) -> list[str]:
    """Dynamically probe model to find symptoms that maximize the target hypothesis."""
    if model is None:
        return []
    
    unselected = [s for s in all_symptoms if s not in current_symptoms]
    final_symptoms = []
    
    base_probs = np.array(base_res["probabilities"])
    sorted_indices = np.argsort(base_probs)[::-1]
    
    current_rank = target_rank
    
    while len(final_symptoms) < top_n and current_rank < len(label_names):
        target_idx = sorted_indices[current_rank]
        base_prob = base_probs[target_idx]
        target_class_name = label_names[target_idx]
        
        # EXCEPTION LOGIC: If the user has already entered all symptoms for the top predicted disease,
        # do not ask disambiguation questions for lower-ranked diseases. Just skip the survey.
        if current_rank == 0:
            try:
                import json, os
                map_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'disease_symptoms.json')
                with open(map_path, 'r') as f:
                    disease_map = json.load(f)
                top_disease_symptoms = disease_map.get(target_class_name, [])
                
                clean_top = set([s.strip().lower() for s in top_disease_symptoms])
                clean_cur = set([s.strip().lower() for s in current_symptoms])
                
                if len(clean_top) > 0 and clean_top.issubset(clean_cur):
                    return []  # Skip survey
            except Exception:
                pass
        
        scores = []
        for sym in unselected:
            if sym in final_symptoms:
                continue
                
            test_syms = current_symptoms + [sym]
            vec = build_symptom_vector(test_syms, all_symptoms)
            res = run_inference(model, vec, img_tensor, label_names)
            
            new_prob = res["probabilities"][target_idx]
            shift = new_prob - base_prob
            
            # Only consider symptoms that POSITIVELY shift the target disease probability
            if shift > 0:
                scores.append((sym, shift))
                
        # Sort symptoms by how much they boost the probability
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for sym, _ in scores:
            if sym not in final_symptoms:
                final_symptoms.append(sym)
            if len(final_symptoms) >= top_n:
                break
                
        # If we found at least some symptoms for the top class, stop. 
        # Don't ask about rank 2 if rank 1 gave us questions.
        if len(final_symptoms) > 0:
            break
            
        current_rank += 1
        
    return final_symptoms


# =============================================================================
# STEP 1 — SYMPTOM INPUT
# =============================================================================

def render_symptoms(all_symptoms: list[str]) -> list[str]:
    st.markdown("""
    <div class="mm-card">
        <h3 style="margin:0 0 .8rem">🩺 Step 1 — Symptom Checklist</h3>
        <p style="color:#64748b;font-size:.9rem;margin:0">
            Describe your condition in plain words <em>and/or</em> select symptoms
            manually below. Both sources are combined for prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Engine badge — shows whether Gemini AI or offline engine is active
    engine_badge = (
        "<span style='background:#7c3aed;color:#fff;border-radius:999px;"
        "padding:.15rem .65rem;font-size:.72rem;font-weight:700;"
        "letter-spacing:.04em;margin-left:.5rem'>✨ AI Powered</span>"
        if _GEMINI_KEY else
        "<span style='background:#0369a1;color:#fff;border-radius:999px;"
        "padding:.15rem .65rem;font-size:.72rem;font-weight:700;"
        "letter-spacing:.04em;margin-left:.5rem'>🧠 Smart Offline</span>"
    )

    st.markdown(
        f"<p style='font-size:.82rem;color:#475569;margin-bottom:.3rem'>"
        f"Describe how you feel in your own words {engine_badge}</p>",
        unsafe_allow_html=True,
    )

    description = st.text_area(
        "💬 Describe your symptoms",
        placeholder=(
            "Describe freely — e.g. 'I have been running temperature since yesterday, "
            "my head is paining me and my body dey weak. I also have a runny nose and "
            "it is hard to breathe properly.' Nigerian English and Pidgin are supported."
        ),
        height=130,
        key="symptom_description",
        help="Write in your own words. The system understands plain English, Nigerian English, and Pidgin.",
    )

    extracted, explanations, engine = extract_symptoms_from_text(description, all_symptoms)

    # ── Show Gemini error as 3-second toast popup ─────────────────────────────
    if st.session_state.get("gemini_error"):
        st.toast(st.session_state.pop("gemini_error"), icon="⚠️")

    if extracted:
        engine_label = "Gemini AI" if engine == "gemini" else "Smart Offline Engine"
        pills = " ".join(
            f"<span style='"
            f"background:rgba(37,99,235,.1);color:#1d4ed8;"
            f"border:1px solid rgba(37,99,235,.25);border-radius:999px;"
            f"padding:.2rem .65rem;font-size:.78rem;font-weight:600;"
            f"margin:.15rem .1rem;display:inline-block'>"
            f"{s.replace('_', ' ')}</span>"
            for s in extracted
        )
        st.markdown(
            f"""
            <div style="margin:.5rem 0 .8rem">
                <p style="color:#64748b;font-size:.82rem;margin:0 0 .4rem">
                    ✨ <strong>{len(extracted)}</strong> symptom(s) detected
                    <span style="color:#94a3b8;font-size:.75rem">
                    via {engine_label}</span>:
                </p>
                <div>{pills}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🔍 How were these detected?", expanded=False):
            for exp in explanations:
                st.markdown(f"- {exp}")
    elif description.strip():
        st.markdown(
            "<div style='background:#fef9c3;border:1px solid #fde047;"
            "border-radius:8px;padding:.6rem .9rem;margin:.3rem 0 .6rem'>"
            "<strong style='color:#854d0e'>⚠️ No symptoms recognised</strong> "
            "<span style='color:#78350f;font-size:.87rem'>— try describing more specifically, "
            "e.g. 'my chest is tight', 'I have been running temperature', "
            "or select symptoms manually below.</span></div>",
            unsafe_allow_html=True,
        )

    search = st.text_input(
        "🔍 Filter symptoms",
        placeholder="Type to search (e.g. 'fever', 'pain') …",
        label_visibility="collapsed",
        key="sym_search_box",
    )

    _MK = "manual_symptom_select"
    current_sel: list = st.session_state.get(_MK, [])
    if search:
        q = search.lower().strip()
        # Replace spaces with underscores so "muscle pain" matches "muscle_pain"
        q_under = q.replace(" ", "_")
        q_words = q.split()

        def _symptom_score(s: str) -> float:
            """Return match score: higher = better match."""
            sl = s.lower()
            # Exact or substring match on underscored form
            if q_under in sl or q in sl:
                return 1.0
            # All query words appear somewhere in the symptom
            if all(w in sl for w in q_words):
                return 0.9
            # Any query word appears in the symptom
            if any(w in sl for w in q_words):
                return 0.7
            # Symptom words appear in query (reverse match: "pain body" finds "body_pain")
            s_words = sl.replace("_", " ").split()
            if any(w in q for w in s_words):
                return 0.5
            return 0.0

        matches = [s for s in all_symptoms if _symptom_score(s) > 0]
        # Sort by score descending
        matches.sort(key=lambda s: _symptom_score(s), reverse=True)
        filtered = list(dict.fromkeys(matches + [s for s in current_sel if s not in matches] + [s for s in extracted if s not in matches]))
    else:
        filtered = all_symptoms

    default_vals = [s for s in extracted if s in filtered]

    with st.form(key="manual_symptoms_form", border=False):
        manual = st.multiselect(
            "Select additional symptoms",
            options=filtered,
            default=default_vals,
            placeholder="Click here to add or remove symptoms …",
            key=_MK,
        )
        st.form_submit_button("✅ Apply Manual Symptoms")

    combined = sorted(set(extracted) | set(manual))

    if combined:
        added = len(set(manual) - set(extracted))
        st.markdown(
            f"<p style='color:#059669;font-size:.88rem'>"
            f"✅ <strong>{len(combined)}</strong> symptom(s) will be used for prediction"
            f"{f' ({len(extracted)} from description, {added} added manually)' if extracted else ''}"
            f"</p>",
            unsafe_allow_html=True,
        )

    return combined


# =============================================================================
# STEP 2 — IMAGE UPLOAD
# =============================================================================

def render_image_upload() -> Optional[object]:
    st.markdown("""
    <div class="mm-card">
        <h3 style="margin:0 0 .8rem">🖼️ Step 2 — Medical Image (Optional)</h3>
        <p style="color:#64748b;font-size:.9rem;margin:0">
            Upload an X-ray or skin scan image. If omitted, a zero-tensor placeholder
            is used and symptom data drives the prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    ver      = st.session_state.get("upload_ver", 0)
    uploaded = st.file_uploader(
        "Upload medical image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key=f"img_upload_{ver}",
        label_visibility="collapsed",
        help="50 MB per file • PNG, JPG, BMP, TIF",
    )

    if uploaded:
        img = Image.open(uploaded)
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            st.image(img, use_container_width=True, caption="Uploaded image")
        with c2:
            st.markdown(
                f"""
                <div style="padding:.6rem 0;color:#0f172a">
                    <span style="color:#64748b">File:</span> {uploaded.name}<br>
                    <span style="color:#64748b">Size:</span> {uploaded.size/1024:.1f} KB<br>
                    <span style="color:#64748b">Format:</span> {img.format}<br>
                    <span style="color:#64748b">Dimensions:</span> {img.width} × {img.height} px
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🗑️ Remove Image", key="btn_remove_img"):
                st.session_state["upload_ver"] = ver + 1
                st.rerun()

    return uploaded


# =============================================================================
# STEP 3 — LOCATION
# =============================================================================

def _reverse_geocode(lat: float, lon: float) -> str:
    """Convert coordinates to a human-readable address via Nominatim (free)."""
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="medimap_ai_v1", timeout=6)
        location = geolocator.reverse((lat, lon), language="en")
        return location.address if location else ""
    except Exception:
        return ""


def render_location() -> tuple[Optional[float], Optional[float], str]:
    st.markdown("""
    <div class="mm-card">
        <h3 style="margin:0 0 .8rem">📍 Step 3 — Your Location</h3>
        <p style="color:#64748b;font-size:.9rem;margin:0 0 1rem">
            Use your device's GPS for the most accurate results, or enter
            your address / coordinates manually.
        </p>
    """, unsafe_allow_html=True)

    method = st.radio(
        "Location method",
        ["📡 GPS", "📝 Address", "🗺️ Coordinates"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

    lat: Optional[float] = None
    lon: Optional[float] = None
    addr: str = ""

    # ── METHOD 1: GPS via browser Geolocation API ─────────────────────────
    if method == "📡 GPS":
        try:
            from streamlit_geolocation import streamlit_geolocation

            st.markdown(
                "<p style='color:#64748b;font-size:.85rem;margin-bottom:.6rem'>"
                "Click the button below — your browser will ask permission to "
                "share your location. <strong>Allow</strong> it for GPS pinpointing.</p>",
                unsafe_allow_html=True,
            )

            gps = streamlit_geolocation()

            if gps and gps.get("latitude") is not None:
                gps_lat = float(gps["latitude"])
                gps_lon = float(gps["longitude"])
                accuracy = gps.get("accuracy")

                st.session_state["gps_lat"] = gps_lat
                st.session_state["gps_lon"] = gps_lon

                acc_colour = "#166534" if accuracy and accuracy < 200 else "#92400e"
                acc_bg     = "#dcfce7" if accuracy and accuracy < 200 else "#fef3c7"
                acc_label  = f"±{accuracy:.0f} m" if accuracy else "unknown"

                cache_key = f"rgc_{gps_lat:.5f}_{gps_lon:.5f}"
                if cache_key not in st.session_state:
                    with st.spinner("Resolving address…"):
                        st.session_state[cache_key] = _reverse_geocode(gps_lat, gps_lon)
                resolved_addr = st.session_state.get(cache_key, "")

                st.markdown(
                    f"""
                    <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                        border-radius:10px;padding:1rem 1.2rem;margin-top:.5rem">
                        <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem">
                            <span style="font-size:1.4rem">📍</span>
                            <strong style="color:#166534;font-size:1rem">Location Detected</strong>
                            <span style="background:{acc_bg};color:{acc_colour};
                                border-radius:999px;padding:.15rem .6rem;
                                font-size:.75rem;font-weight:600">
                                Accuracy {acc_label}
                            </span>
                        </div>
                        <table style="border:none;width:100%;font-size:.88rem">
                            <tr>
                                <td style="color:#64748b;padding:.15rem .4rem .15rem 0;
                                    border:none;width:90px">Latitude</td>
                                <td style="color:#0f172a;font-weight:600;
                                    border:none">{gps_lat:.6f}°</td>
                            </tr>
                            <tr>
                                <td style="color:#64748b;padding:.15rem .4rem .15rem 0;
                                    border:none">Longitude</td>
                                <td style="color:#0f172a;font-weight:600;
                                    border:none">{gps_lon:.6f}°</td>
                            </tr>
                            {"<tr><td style='color:#64748b;padding:.15rem .4rem .15rem 0;border:none;vertical-align:top'>Address</td>"
                             f"<td style='color:#0f172a;border:none'>{resolved_addr}</td></tr>"
                             if resolved_addr else ""}
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                lat = gps_lat
                lon = gps_lon

            elif st.session_state.get("gps_lat") is not None:
                lat = st.session_state["gps_lat"]
                lon = st.session_state["gps_lon"]
                st.info(f"📍 Using last GPS fix: {lat:.5f}°, {lon:.5f}°")

        except ImportError:
            st.warning("streamlit-geolocation is not installed.")

    # ── METHOD 2: Free-text address ───────────────────────────────────────
    elif method == "📝 Address":
        addr = st.text_input(
            "Address / City / Plus Code",
            placeholder="e.g. Abuja, Nigeria  |  5M23+GR3 Zaria, Kaduna  |  6 Ahmadu Bello Way Lagos",
            key="manual_address",
        )
        if addr.strip():
            st.caption(
                "✅ Supports: city names · street addresses · "
                "**Google Plus Codes** (e.g. `5M23+GR3 Zaria`) — "
                "geocoded via Geoapify + OSM with Nigeria coverage."
            )

    # ── METHOD 3: Raw coordinates ─────────────────────────────────────────
    else:  # "🗺️ Coordinates"
        c1, c2 = st.columns(2, gap="small")
        with c1:
            lat = st.number_input(
                "Latitude", value=51.5074, format="%.6f",
                min_value=-90.0, max_value=90.0, key="coord_lat",
            )
        with c2:
            lon = st.number_input(
                "Longitude", value=-0.1278, format="%.6f",
                min_value=-180.0, max_value=180.0, key="coord_lon",
            )
        st.caption("✅ Manual coordinates will be used directly.")

    return lat, lon, addr


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    # ── Session state defaults ────────────────────────────────────────────
    st.session_state.setdefault("result",     None)
    st.session_state.setdefault("analysis",   None)
    st.session_state.setdefault("upload_ver", 0)

    # ── Sidebar ───────────────────────────────────────────────────────────
    config = render_sidebar()

    # ── Style ─────────────────────────────────────────────────────────────
    inject_style()

    # ── Hero ──────────────────────────────────────────────────────────────
    render_hero()

    # ── Load model resources ──────────────────────────────────────────────
    symptom_cols  = load_symptom_columns()
    label_encoder = load_label_encoder()
    label_names   = (list(label_encoder.classes_)
                     if label_encoder else ALL_DISEASES)
    model = load_model(len(symptom_cols), len(label_names), config["backbone"])

    if model is None:
        st.info(
            "🛠️ **Demo Mode** — No trained model checkpoint found. "
            "The system will return illustrative predictions. "
            "Train the model using `python models/train.py` to enable real inference."
        )

    # ── Input columns ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        selected_symptoms = render_symptoms(symptom_cols)
        uploaded_file     = render_image_upload()

    with col_right:
        user_lat, user_lon, user_address = render_location()

        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button(
            "🔬 Analyse & Find Specialists",
            use_container_width=True,
            key="btn_analyse",
        )

    # ── Prediction Processor ──────────────────────────────────────────────
    def _process_prediction(final_result, final_syms):
        geo_result = None
        try:
            geo_result = recommend_specialists(
                disease      = final_result["predicted_class"],
                user_address = user_address or None,
                user_lat     = user_lat,
                user_lon     = user_lon,
                radius_km    = config["radius_km"],
                max_results  = config["max_clinics"],
                map_tiles    = config["map_theme"],
            )
        except (RuntimeError, ValueError) as geo_err:
            logger.warning("GIS lookup failed: %s", geo_err)
            st.warning(
                f"⚠️ Clinic map unavailable: **{geo_err}**  "
                "Your diagnosis is still ready — view it on the Analysis page."
            )

        st.session_state["result"]   = final_result
        st.session_state["analysis"] = {
            "result"     : final_result,
            "clinics"    : [c.to_dict() for c in geo_result["clinics"]] if geo_result else [],
            "fmap"       : geo_result["map"] if geo_result and geo_result.get("map") else None,
            "map_html"   : geo_result["map"]._repr_html_() if geo_result and geo_result.get("map") else None,
            "specialty"  : geo_result["specialty"] if geo_result else "",
            "config"     : config,
            "label_names": label_names,
            "final_symptoms": final_syms,
            "description": st.session_state.get("symptom_description", "")
        }
        st.switch_page("pages/2_Analysis.py")

    # ── Analysis trigger ──────────────────────────────────────────────────
    if analyse_btn:
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom or describe your condition.")
            st.stop()

        st.session_state["confirmed_survey_symptoms"] = []
        st.session_state["survey_target_rank"] = 0
        st.session_state["analyse_active"] = True

    if st.session_state.get("analyse_active", False):
        with st.spinner("Running analysis …"):
            final_symptoms = selected_symptoms + st.session_state.get("confirmed_survey_symptoms", [])
            sym_vec = build_symptom_vector(final_symptoms, symptom_cols)
            img_ten = (preprocess_image(uploaded_file)
                       if uploaded_file else torch.zeros(3, 224, 224))

            result = run_inference(model, sym_vec, img_ten, label_names)
            
            target_rank = st.session_state.get("survey_target_rank", 0)

            # If confidence is low and we haven't exhausted the top 5 diseases
            if result["confidence"] < 60.0 and not result.get("demo", False) and target_rank < 5:
                questions = get_clarifying_questions(
                    model, final_symptoms, symptom_cols, label_names, result, img_ten, top_n=10, target_rank=target_rank
                )
                if questions:
                    st.session_state["survey_mode"] = True
                    st.session_state["base_result"] = result
                    st.session_state["survey_questions"] = questions
                else:
                    # No more symptoms to ask about, just accept the low confidence and proceed
                    st.session_state["survey_mode"] = False
                    st.session_state["analyse_active"] = False
                    _process_prediction(result, final_symptoms)
                    st.rerun()
            else:
                st.session_state["survey_mode"] = False
                st.session_state["analyse_active"] = False
                _process_prediction(result, final_symptoms)
                st.rerun()

    # ── Survey UI ─────────────────────────────────────────────────────────
    if st.session_state.get("survey_mode"):
        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#fef9c3;border:1px solid #fde047;border-radius:8px;padding:1rem'>"
            "<h4 style='margin-top:0;color:#854d0e'>🩺 Diagnostic Follow-up Required</h4>"
            "<p style='color:#78350f;margin-bottom:0.5rem'>"
            f"The current symptoms lack specificity (Confidence: <b>{st.session_state['base_result']['confidence']:.1f}%</b>). "
            f"The system currently suspects <b>{st.session_state['base_result']['predicted_class']}</b>. "
            "To confirm or narrow down the diagnosis, please check any of the following you are experiencing. "
            "<i>The system will automatically re-evaluate upon every click!</i></p></div>",
            unsafe_allow_html=True
        )
        
        with st.form(key="survey_form"):
            st.markdown("<b>Are you experiencing any of these?</b>", unsafe_allow_html=True)
            
            survey_answers = []
            for sym in st.session_state.get("survey_questions", []):
                if st.checkbox(sym.replace("_", " ").title(), key=f"survey_checkbox_{sym}"):
                    survey_answers.append(sym)
                    
            c1, c2 = st.columns(2)
            with c1:
                confirm_clicked = st.form_submit_button("✅ Confirm Checked", type="primary", use_container_width=True)
            with c2:
                none_clicked = st.form_submit_button("❌ None of these", use_container_width=True)

        if confirm_clicked:
            if survey_answers:
                st.session_state["confirmed_survey_symptoms"].extend(survey_answers)
                st.session_state["survey_target_rank"] = 0
                st.rerun()
            else:
                st.warning("Please check at least one box, or click 'None of these'.")
                
        if none_clicked:
            st.session_state["survey_target_rank"] = st.session_state.get("survey_target_rank", 0) + 1
            st.rerun()

    elif st.session_state.get("analysis"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "📊 Analysis results are ready. "
            "Click below to view your full diagnostic report and hospital map."
        )
        if st.button("📈 View Analysis Results", use_container_width=True):
            st.switch_page("pages/2_Analysis.py")

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#64748b;font-size:.8rem'>"
        "MediMap AI © 2024 | Built with PyTorch, Streamlit &amp; OpenStreetMap | "
        "<span style='color:#ef4444'>NOT a medical device</span> "
        "— for research &amp; educational use only.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

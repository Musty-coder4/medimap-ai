"""
MediMap AI — Page 2: Analysis Results
======================================
Displays the full diagnostic report, probability chart, interactive
hospital map, and specialist clinic table.

This page reads from st.session_state["analysis"] which is populated
by the main input page after a successful inference run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import os
import numpy as np
import pandas as pd
import streamlit as st
from google import genai

try:
    from streamlit_folium import st_folium
    _HAS_ST_FOLIUM = True
except ImportError:
    import streamlit.components.v1 as components
    _HAS_ST_FOLIUM = False

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Import project utils (must be after sys.path fix) ────────────────────────
from utils.disease_insights_library import get_fallback_insights

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediMap AI — Analysis",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='margin:0 0 1rem;font-size:1.2rem'>&#x2699;&#xFE0F; Navigation</h2>",
        unsafe_allow_html=True,
    )
    st.divider()
    if st.button("&#x2190; New Analysis", use_container_width=True, key="btn_back"):
        st.switch_page("main.py")

# ── Fixed light-mode stylesheet ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
[data-testid="block-container"],.main .block-container {
    background-color:#f8fafc!important;
    font-family:'Inter',sans-serif!important;
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"]>div,
section[data-testid="stSidebar"]>div:first-child {
    background-color:#f1f5f9!important;
    border-right:1px solid #e2e8f0!important;
}
html,body,.stApp,.stMarkdown,.stText,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
.stMarkdown p,.stMarkdown span,
p,span,label,li,td,th,h1,h2,h3,h4,h5,h6 { color:#0f172a!important; }

.stSelectbox label,.stMultiSelect label,.stSlider label,.stRadio label,
.stCheckbox label,.stToggle label,.stTextInput label,.stTextArea label,
.stFileUploader label,[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p { color:#0f172a!important; }

.stCaption,[data-testid="stCaptionContainer"],.stCaption p,small { color:#64748b!important; }

[data-testid="metric-container"] {
    background-color:#ffffff!important;
    border:1px solid #e2e8f0!important;
    border-radius:12px!important;
    box-shadow:0 2px 12px rgba(0,0,0,.08)!important;
}
[data-testid="stMetricValue"]         { color:#2563eb!important; }
[data-testid="stMetricLabel"] p       { color:#64748b!important; }

input,textarea,select {
    background-color:#ffffff!important;
    color:#0f172a!important;
    border-color:#e2e8f0!important;
}
[data-baseweb="input"],[data-baseweb="input"]>div,
[data-baseweb="base-input"],[data-baseweb="base-input"]>div,
[data-baseweb="base-input"] input,
[data-baseweb="textarea"],[data-baseweb="textarea"]>div,
[data-baseweb="textarea"] textarea {
    background-color:#ffffff!important;
    color:#0f172a!important;
    border-color:#e2e8f0!important;
}
[data-baseweb="select"],[data-baseweb="select"]>div {
    background-color:#ffffff!important;
    border-color:#e2e8f0!important;
}
[data-baseweb="select"] [data-baseweb="single-value"],
[data-baseweb="select"] span[class*="singleValue"] { color:#0f172a!important; }
[data-baseweb="select"] [data-baseweb="placeholder"],
[data-baseweb="select"] span[class*="placeholder"]  { color:#64748b!important; }
[data-baseweb="tag"] {
    background-color:#2563eb!important;
    color:#fff!important;
    border-radius:6px!important;
}
[data-baseweb="tag"] span { color:#fff!important; }
[data-baseweb="popover"],[data-baseweb="menu"],
ul[role="listbox"],[role="listbox"] {
    background-color:#ffffff!important;
    border:1px solid #e2e8f0!important;
}
[role="option"],[data-baseweb="menu-item"],li[role="option"] {
    background-color:#ffffff!important;
    color:#0f172a!important;
}
[role="option"]:hover,[data-baseweb="menu-item"]:hover { background-color:#f1f5f9!important; }

[data-testid="stAlert"]>div,.stAlert {
    background-color:#ffffff!important;
    border:1px solid #e2e8f0!important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] code { color:#0f172a!important; }

.stButton>button {
    background:linear-gradient(135deg,#2563eb,#6366f1)!important;
    color:#fff!important;border:none!important;border-radius:10px!important;
    font-weight:600!important;transition:opacity .2s,transform .15s!important;
}
.stButton>button:hover { opacity:.92!important;transform:translateY(-1px)!important; }

.stDataFrame { border-radius:10px;overflow:hidden; }
[data-testid="stDataFrameResizable"] { background-color:#ffffff!important; }

hr { border-color:#e2e8f0!important; }

.mm-card {
    background-color:#ffffff!important;
    border:1px solid #e2e8f0!important;
    border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1.2rem;
    box-shadow:0 2px 12px rgba(0,0,0,.08);color:#0f172a!important;
}
.mm-card h3,.mm-card p,.mm-card span { color:#0f172a!important; }

.badge {
    border-radius:999px;padding:.25rem .75rem;
    font-size:.78rem;font-weight:600;display:inline-block;
}
.badge-green { background:#dcfce7;color:#166534; }
.badge-warn  { background:#fef3c7;color:#92400e; }
.badge-blue  { background:#dbeafe;color:#1e40af; }
</style>
""", unsafe_allow_html=True)

# ── Read analysis data from session state ────────────────────────────────────
analysis = st.session_state.get("analysis")

if not analysis:
    st.markdown("""
    <div class="mm-card" style="text-align:center;padding:3rem">
        <h2>📋 No Analysis Yet</h2>
        <p style="color:#64748b;margin:1rem 0">
            Run a diagnosis on the input page first.
        </p>
    </div>""", unsafe_allow_html=True)
    if st.button("← Back to Input", use_container_width=False):
        st.switch_page("main.py")
    st.stop()

result      = analysis["result"]
clinics     = analysis["clinics"]       # list of dicts
map_html    = analysis["map_html"]      # folium HTML string or None
specialty   = analysis["specialty"]
config      = analysis["config"]
label_names = analysis["label_names"]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin:0'>&#x1F4CA; Diagnostic Analysis Report</h1>",
    unsafe_allow_html=True,
)
st.divider()

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4, gap="medium")
m1.metric("🔬 Predicted Disease", result["predicted_class"])
m2.metric("📊 Confidence",        f"{result['confidence']:.1f}%")
m3.metric("🏥 Specialist",        specialty or "—")
m4.metric("📍 Clinics Found",     len(clinics))

st.divider()

# ── Diagnosis card ────────────────────────────────────────────────────────────
conf = result["confidence"]
if conf >= 70:
    badge_cls, badge_lbl = "badge-green", "High Confidence"
elif conf >= 45:
    badge_cls, badge_lbl = "badge-warn",  "Moderate Confidence"
else:
    badge_cls, badge_lbl = "badge-blue",  "Low Confidence"

st.markdown(
    f"""
    <div class="mm-card" style="border-left:4px solid #2563eb">
        <div style="display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;gap:1rem">
            <div>
                <p style="color:#64748b;font-size:.8rem;margin:0;
                    letter-spacing:.1em;text-transform:uppercase">Predicted Condition</p>
                <h2 style="margin:.3rem 0;font-size:1.9rem">{result['predicted_class']}</h2>
                <p style="color:#64748b;margin:0">
                    Recommended specialist:
                    <strong style="color:#059669">{specialty}</strong>
                </p>
            </div>
            <div style="text-align:right">
                <span class="badge {badge_cls}" style="font-size:.85rem">{badge_lbl}</span>
                <div style="font-size:2.4rem;font-weight:700;color:#2563eb;margin-top:.4rem">
                    {conf:.1f}%
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input Profile ─────────────────────────────────────────────────────────────
final_syms  = analysis.get("final_symptoms", [])
description = analysis.get("description", "")

if final_syms or description:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        pills = " ".join(
            f"<span style='background:rgba(37,99,235,.1);color:#1d4ed8;"
            f"border:1px solid rgba(37,99,235,.25);border-radius:999px;"
            f"padding:.2rem .65rem;font-size:.78rem;font-weight:600;"
            f"margin:.15rem .1rem;display:inline-block'>{s.replace('_', ' ').title()}</span>"
            for s in final_syms
        ) if final_syms else "<span style='color:#64748b'>None</span>"
        
        st.markdown(
            f"""
            <div class="mm-card" style="height: 100%;">
                <h4 style="margin:0 0 .8rem;color:#0f172a">🩺 Analyzed Symptoms</h4>
                <div>{pills}</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        desc_html = f"<p style='color:#475569;font-style:italic'>\"{description}\"</p>" if description.strip() else "<p style='color:#64748b'>No text description provided.</p>"
        st.markdown(
            f"""
            <div class="mm-card" style="height: 100%;">
                <h4 style="margin:0 0 .8rem;color:#0f172a">💬 Patient Description</h4>
                {desc_html}
            </div>
            """, unsafe_allow_html=True
        )

# ── Generative AI Insights ───────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

@st.cache_data(show_spinner=False, ttl=3600)
def get_disease_insights(disease_name: str, key: str) -> str | None:
    if not key:
        return None
    try:
        client = genai.Client(api_key=key)
        prompt = (
            f"You are a helpful medical assistant. The user's AI model just diagnosed them with {disease_name}. "
            "Provide a 2-sentence plain-english description of what this disease is, followed by a list of 4 "
            "immediate, safe at-home precautions they can take before seeing a specialist. Format the output cleanly "
            "with markdown. Do not include introductory filler text."
        )
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error generating insights: {e}"

st.markdown("<h3 style='margin-top:2rem'>✨ AI Health Insights</h3>", unsafe_allow_html=True)

gemini_ok = False
if _GEMINI_KEY:
    with st.spinner("Generative AI is typing your custom health insights..."):
        insights = get_disease_insights(result['predicted_class'], _GEMINI_KEY)

    if insights and not insights.startswith("⚠️"):
        gemini_ok = True
        st.markdown('<div class="mm-card" style="border-top:4px solid #8b5cf6">', unsafe_allow_html=True)
        st.markdown(insights)
        st.markdown('</div>', unsafe_allow_html=True)

if not gemini_ok:
    # Fallback to static library
    fallback = get_fallback_insights(result['predicted_class'])
    if fallback:
        st.markdown(
            '<div class="mm-card" style="border-top:4px solid #8b5cf6">',
            unsafe_allow_html=True
        )
        st.markdown(f"{fallback['description']}")
        st.markdown("**What you can do at home:**")
        for item in fallback['precautions']:
            st.markdown(f"- {item}")
        st.markdown('</div>', unsafe_allow_html=True)
        if _GEMINI_KEY:
            st.caption("ℹ️ Showing offline health insights — Gemini AI was temporarily unavailable.")
        else:
            st.caption("ℹ️ Showing offline health insights. Add a GEMINI_API_KEY to your .env for dynamic AI-powered insights.")
    else:
        st.warning("No health insights available for this condition.")

# ── Probability chart ─────────────────────────────────────────────────────────
if config.get("show_chart", True) and result.get("probabilities") is not None:
    import plotly.graph_objects as go

    probs      = result["probabilities"]
    top_idx    = np.argsort(probs)[::-1][:10]
    top_labels = [label_names[i] if label_names else str(i) for i in top_idx]
    top_probs  = [probs[i] * 100 for i in top_idx]
    colours    = ["#2563eb" if l == result["predicted_class"] else "#cbd5e1"
                  for l in top_labels]

    fig = go.Figure(go.Bar(
        x=top_probs[::-1], y=top_labels[::-1], orientation="h",
        marker_color=colours[::-1],
        text=[f"{p:.1f}%" for p in top_probs[::-1]], textposition="outside",
    ))
    fig.update_layout(
        title="Top-10 Disease Probability Distribution",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Inter, sans-serif"),
        xaxis=dict(title="Confidence (%)", gridcolor="#e2e8f0",
                   range=[0, max(top_probs) * 1.18]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=400, margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── GIS Map + Clinic table ────────────────────────────────────────────────────
badge_count = (
    f"<span class='badge badge-blue' style='font-size:.75rem'>{len(clinics)} results</span>"
    if clinics else
    "<span class='badge badge-warn' style='font-size:.75rem'>No results</span>"
)
st.markdown(
    f"<h3 style='margin-bottom:.8rem'>🗺️ Nearby {specialty} Clinics {badge_count}</h3>",
    unsafe_allow_html=True,
)

if map_html:
    map_col, tbl_col = st.columns([3, 2], gap="medium")

    with map_col:
        # We use standard HTML iframe component instead of st_folium because 
        # st_folium has known bugs causing blank white space inside st.columns.
        if map_html:
            import streamlit.components.v1 as components
            # The Folium HTML is self-contained. Render it directly.
            components.html(map_html, height=520, scrolling=False)
        else:
            st.warning("Map could not be generated.")

    with tbl_col:
        if clinics:
            st.caption("🔢 Numbered markers on the map match rows below.")
            df   = pd.DataFrame(clinics)
            cols = ["Name", "Specialty", "Distance (km)", "Address", "Phone", "Website"]
            df   = df[[c for c in cols if c in df.columns]]
            st.dataframe(
                df.style.format({"Distance (km)": "{:.2f}"}),
                width="stretch",
                height=490,
                hide_index=True,
            )
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Clinic List (CSV)",
                csv,
                file_name=f"{specialty.replace(' ', '_')}_clinics.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No named clinics found in the search radius. "
                    "Try increasing the radius in the sidebar.")

elif clinics:
    st.info("🗺️ Map could not be rendered. Clinic list is shown below.")
    df = pd.DataFrame(clinics)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.markdown(
        "<div class='mm-card' style='text-align:center;padding:2rem'>"
        "<p style='color:#64748b'>📍 No specialist clinics found. "
        "Try a different location or increase the search radius in the sidebar.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='color:#64748b;font-size:.8rem;text-align:center'>"
    "⚕️ MediMap AI — For informational purposes only. "
    "Always consult a qualified medical professional.</p>",
    unsafe_allow_html=True,
)

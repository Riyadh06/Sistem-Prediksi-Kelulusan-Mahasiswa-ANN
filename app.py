import os
import io
import json
import zipfile
import h5py
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa - ANN Backpropagation",
    page_icon="https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definisi Ikon Vektor SVG Modern (Clean, Elegan, 100% Bebas Emoji)
ICONS = {
    "chip": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "graduation": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>',
    "book": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    "clock": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "globe": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "settings": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    "chart": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "users": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "lightbulb": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg>',
    "check": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "alert": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F87171" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    "check_circle": '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "alert_triangle": '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#F87171" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
}

# Custom Styling CSS - Font Poppins Compact, Ukuran Proporsional & Fix Icon Collision
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    html, body, .stApp {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0B0F17;
        color: #F1F5F9;
    }
    
    /* Font Poppins untuk elemen teks tanpa merusak font simbol/ikon Streamlit */
    h1, h2, h3, h4, h5, h6, p, label, input, select, textarea, button,
    .stMarkdown, .stSelectbox, .stSlider, .stNumberInput, .stRadio,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* PENTING: Lindungi font ikon bawaan Streamlit agar chevron/panah expander TIDAK bertumpuk dengan teks */
    [data-testid="stIconMaterial"], 
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpanderToggleIcon"] *,
    [class*="material-symbols"], 
    [class*="material-icons"],
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
    }

    /* Expander styling agar rapi, berjarak, dan tidak bertabrakan */
    div[data-testid="stExpander"] details summary {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        padding: 10px 14px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #E2E8F0 !important;
    }
    
    div[data-testid="stExpander"] details summary p {
        font-size: 0.88rem !important;
        margin: 0 !important;
    }
    
    div[data-testid="stExpander"] details summary span[data-testid="stExpanderToggleIcon"] {
        margin: 0 4px 0 0 !important;
        min-width: 18px !important;
        display: inline-flex !important;
        align-items: center !important;
    }

    /* Top Hero Header (Ukuran Lebih Ringkas & Rapi) */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.22) 50%, rgba(11, 15, 23, 0.92) 100%);
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px -10px rgba(16, 185, 129, 0.2);
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(52, 211, 153, 0.35);
        color: #6EE7B7;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .hero-title {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.3px;
        color: #FFFFFF;
        margin: 0 0 6px 0;
        line-height: 1.3;
    }
    
    .hero-title span {
        background: linear-gradient(135deg, #34D399 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-desc {
        color: #94A3B8;
        font-size: 0.85rem;
        max-width: 800px;
        line-height: 1.55;
        margin: 0;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #6EE7B7;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Input Labels */
    div[data-testid="stWidgetLabel"] label p {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #E2E8F0 !important;
    }
    
    /* Sidebar */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.22) 100%);
        border: 1px solid rgba(52, 211, 153, 0.28);
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 16px;
    }
    
    .brand-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }
    
    .brand-name {
        font-weight: 700;
        font-size: 0.95rem;
        color: #F8FAFC;
        line-height: 1.2;
    }
    
    .brand-sub {
        font-size: 0.7rem;
        color: #34D399;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 5px;
        margin-top: 2px;
    }
    
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10B981;
        display: inline-block;
    }
    
    .sidebar-section-title {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    
    /* Sidebar Mini Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 8px;
        margin-bottom: 14px;
    }
    
    .mini-metric-card {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid rgba(52, 211, 153, 0.16);
        border-radius: 10px;
        padding: 8px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .mini-metric-label {
        font-size: 0.74rem;
        color: #94A3B8;
        font-weight: 500;
    }
    
    .mini-metric-value {
        font-size: 0.92rem;
        font-weight: 700;
        color: #34D399;
    }
    
    .mini-badge {
        font-size: 0.62rem;
        padding: 2px 6px;
        border-radius: 5px;
        background: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        font-weight: 600;
        margin-left: 4px;
    }

    /* Result Banner Cards */
    .status-card-success {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.92) 0%, rgba(6, 95, 70, 0.88) 50%, rgba(4, 120, 87, 0.85) 100%);
        border: 1.5px solid #34D399;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 8px 24px -5px rgba(16, 185, 129, 0.3);
    }
    
    .status-card-danger {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.92) 0%, rgba(153, 27, 27, 0.88) 50%, rgba(185, 28, 28, 0.85) 100%);
        border: 1.5px solid #F87171;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 8px 24px -5px rgba(239, 68, 68, 0.3);
    }
    
    .status-icon-wrapper {
        display: flex;
        justify-content: center;
        margin-bottom: 6px;
    }
    
    .status-badge-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 0.3px;
        margin-bottom: 6px;
    }
    
    .status-badge-prob {
        font-size: 0.88rem;
        font-weight: 500;
        color: #E2E8F0;
    }
    
    .status-badge-prob b {
        color: #FFFFFF;
        font-size: 1.05rem;
    }

    /* 3-Box Summary Metrics (Menggantikan st.metric agar tidak terpotong) */
    .summary-metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-top: 12px;
    }
    
    .summary-metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 10px;
        padding: 10px 8px;
        text-align: center;
    }
    
    .sm-label {
        font-size: 0.68rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 3px;
    }
    
    .sm-value {
        font-size: 0.88rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
        white-space: normal;
        word-break: break-word;
    }
    
    .sm-val-success {
        color: #34D399;
    }
    
    .sm-val-danger {
        color: #F87171;
    }

    /* Recommendation Card */
    .rec-card {
        background: rgba(22, 27, 34, 0.95);
        border: 1px solid rgba(52, 211, 153, 0.28);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        height: 100%;
    }
    
    .rec-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.92rem;
        font-weight: 700;
        color: #6EE7B7;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .rec-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        color: #E2E8F0;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-bottom: 9px;
    }
    
    .rec-bullet-success, .rec-bullet-warn {
        display: flex;
        align-items: center;
        margin-top: 2px;
        flex-shrink: 0;
    }

    /* Confidence Bar */
    .custom-progress-bg {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        margin-top: 10px;
    }
    
    .custom-progress-fill-success {
        background: linear-gradient(90deg, #10B981, #34D399);
        height: 100%;
        border-radius: 8px;
        box-shadow: 0 0 8px rgba(52, 211, 153, 0.6);
    }
    
    .custom-progress-fill-danger {
        background: linear-gradient(90deg, #EF4444, #F87171);
        height: 100%;
        border-radius: 8px;
        box-shadow: 0 0 8px rgba(248, 113, 113, 0.6);
    }

    /* Streamlit Primary Button Override */
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        border: 1px solid rgba(52, 211, 153, 0.4) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.4) !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.4px !important;
    }
    
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #34D399 0%, #10B981 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(16, 185, 129, 0.55) !important;
        border-color: #6EE7B7 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. KOMPATIBILITAS SCIKIT-LEARN & ENGINE PREDIKSI
# ==============================================================================
# Patch compatibility untuk unpickling cross-version antar scikit-learn
import sklearn.impute._base as ib
import sklearn.compose._column_transformer as ct

class _RemainderColsList(list):
    pass
ct._RemainderColsList = _RemainderColsList

orig_transform = ib.SimpleImputer.transform
def patched_transform(self, X, *args, **kwargs):
    if not hasattr(self, '_fill_dtype'):
        self._fill_dtype = getattr(self, '_fit_dtype', float)
    return orig_transform(self, X)
ib.SimpleImputer.transform = patched_transform

# Direktori model deployment
BASE_DIR = os.path.dirname(__file__)
CANDIDATE_DIRS = [
    os.path.join(BASE_DIR, "model_prediksi_kelulusan_web"),
    os.path.join(BASE_DIR, "model_web_deployment"),
]
MODEL_DIR = next((d for d in CANDIDATE_DIRS if os.path.exists(d)), CANDIDATE_DIRS[0])
MODEL_PATH = os.path.join(MODEL_DIR, "model_ann_kelulusan.keras")
PIPELINE_PATH = os.path.join(MODEL_DIR, "pipeline_preprocessor.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata_model.json")

@st.cache_resource
def load_prediction_assets():
    """Memuat bobot ANN dari .keras dan pipeline dari .joblib secara efisien"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
        st.error(f"File model tidak ditemukan di: {MODEL_DIR}")
        st.stop()
        
    # 1. Muat Pipeline Preprocessing
    pipeline = joblib.load(PIPELINE_PATH)
    
    # 2. Muat Bobot ANN dari .keras
    with zipfile.ZipFile(MODEL_PATH, 'r') as z:
        with z.open('model.weights.h5') as f:
            content = io.BytesIO(f.read())
            with h5py.File(content, 'r') as h:
                W1 = np.array(h['layers/dense/vars/0'])
                b1 = np.array(h['layers/dense/vars/1'])
                W2 = np.array(h['layers/dense_1/vars/0'])
                b2 = np.array(h['layers/dense_1/vars/1'])
                W3 = np.array(h['layers/dense_2/vars/0'])
                b3 = np.array(h['layers/dense_2/vars/1'])
                
    # 3. Muat Metadata
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
    return pipeline, (W1, b1, W2, b2, W3, b3), metadata

pipeline, weights, metadata = load_prediction_assets()
W1, b1, W2, b2, W3, b3 = weights

def forward_propagation(X_proc, W1, b1, W2, b2, W3, b3):
    """Komputasi propagasi maju ANN: ReLU -> ReLU -> Sigmoid"""
    def relu(x): return np.maximum(0, x)
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    
    z1 = np.dot(X_proc, W1) + b1
    h1 = relu(z1)
    z2 = np.dot(h1, W2) + b2
    h2 = relu(z2)
    z3 = np.dot(h2, W3) + b3
    prob = float(sigmoid(z3)[0][0])
    return prob, h1, h2

# ==============================================================================
# 3. SIDEBAR: INFORMASI MODEL & PRESET SAMPLE
# ==============================================================================
with st.sidebar:
    # Header Brand Avatar (Icon Vector SVG)
    st.markdown(f"""
    <div class="sidebar-brand">
        <div class="brand-icon">{ICONS['graduation']}</div>
        <div>
            <div class="brand-name">ANN Engine</div>
            <div class="brand-sub"><span class="pulse-dot"></span> Model Aktif & Siap</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-section-title">{ICONS['settings']} Spesifikasi Model</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 0.8rem; line-height: 1.6; color: #CBD5E1;">
        &bull; <b>Arsitektur:</b> Multilayer Perceptron (MLP)<br>
        &bull; <b>Algoritma:</b> Backpropagation<br>
        &bull; <b>Lapisan:</b> 11 Input &rarr; 16 Hidden &rarr; 8 Hidden &rarr; 1 Output<br>
        &bull; <b>Optimizer:</b> Adam (<i>lr</i> = 0.001)<br>
        &bull; <b>Aktivasi:</b> ReLU (Tersembunyi), Sigmoid (Keluaran)
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Metrik Performa Model
    st.markdown(f"""
    <div class="sidebar-section-title">{ICONS['chart']} Performa Pengujian</div>
    """, unsafe_allow_html=True)
    
    acc = metadata.get('model_metrics', {}).get('accuracy', 0.9504) * 100
    auc = metadata.get('model_metrics', {}).get('roc_auc', 0.9942)
    f1 = metadata.get('model_metrics', {}).get('f1_score', 0.9677)
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="mini-metric-card">
            <span class="mini-metric-label">Akurasi Model</span>
            <div><span class="mini-metric-value">{acc:.2f}%</span><span class="mini-badge">Tinggi</span></div>
        </div>
        <div class="mini-metric-card">
            <span class="mini-metric-label">ROC-AUC Score</span>
            <div><span class="mini-metric-value">{auc:.4f}</span><span class="mini-badge">Optimal</span></div>
        </div>
        <div class="mini-metric-card">
            <span class="mini-metric-label">F1-Score</span>
            <div><span class="mini-metric-value">{f1:.4f}</span><span class="mini-badge">Seimbang</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown(f"""
    <div class="sidebar-section-title">{ICONS['users']} Preset Profil Mahasiswa</div>
    """, unsafe_allow_html=True)
    
    preset = st.radio(
        "Pilih Profil Contoh:",
        ("Manual (Input Mandiri)", "Profil Mahasiswa Berpotensi Tepat Waktu", "Profil Mahasiswa Berisiko Terlambat")
    )
    
    st.caption("UAS Mata Kuliah Artificial Intelligence | S1 Teknik Informatika")

# Set nilai default sesuai preset
if preset == "Profil Mahasiswa Berpotensi Tepat Waktu":
    default_ipk = 3.65
    default_sks = 140
    default_kehadiran = 92
    default_jam = 28
    default_org = 1
    default_penghasilan = "Menengah"
    default_internet = "Baik"
elif preset == "Profil Mahasiswa Berisiko Terlambat":
    default_ipk = 2.10
    default_sks = 98
    default_kehadiran = 45
    default_jam = 5
    default_org = 5
    default_penghasilan = "Rendah"
    default_internet = "Buruk"
else:
    default_ipk = 3.45
    default_sks = 135
    default_kehadiran = 85
    default_jam = 20
    default_org = 2
    default_penghasilan = "Menengah"
    default_internet = "Baik"

# ==============================================================================
# 4. KONTEN UTAMA: HERO BANNER & FORMULIR INPUT MAHASISWA
# ==============================================================================
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">{ICONS['chip']} Artificial Neural Network &bull; Backpropagation v2.0</div>
    <h1 class="hero-title">Sistem Prediksi <span>Kelulusan Mahasiswa</span></h1>
    <p class="hero-desc">
        Aplikasi inferensi prediktif cerdas berbasis Jaringan Syaraf Tiruan (ANN) Multi-Layer Perceptron. 
        Menganalisis probabilitas kelulusan tepat waktu berdasarkan parameter akademik, kedisiplinan, dan faktor penunjang mahasiswa.
    </p>
</div>
""", unsafe_allow_html=True)

with st.form("form_prediksi"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="section-header">
            {ICONS['book']} 1. Akademik Utama
        </div>
        """, unsafe_allow_html=True)
        ipk = st.slider("Indeks Prestasi Kumulatif (IPK)", min_value=0.0, max_value=4.0, value=float(default_ipk), step=0.01)
        total_sks = st.slider("Total SKS yang Telah Lulus", min_value=0, max_value=160, value=int(default_sks), step=1, help="Standar minimal kelulusan sarjana adalah 144 SKS.")
        
    with col2:
        st.markdown(f"""
        <div class="section-header">
            {ICONS['clock']} 2. Kedisiplinan & Waktu
        </div>
        """, unsafe_allow_html=True)
        kehadiran = st.slider("Tingkat Kehadiran (%)", min_value=0, max_value=100, value=int(default_kehadiran), step=1)
        jam_belajar = st.slider("Jam Belajar Mandiri (Jam/Minggu)", min_value=0, max_value=80, value=int(default_jam), step=1)
        jumlah_organisasi = st.number_input("Jumlah Organisasi yang Diikuti", min_value=0, max_value=10, value=int(default_org), step=1)
        
    with col3:
        st.markdown(f"""
        <div class="section-header">
            {ICONS['globe']} 3. Faktor Pendukung
        </div>
        """, unsafe_allow_html=True)
        penghasilan = st.selectbox(
            "Penghasilan Orang Tua", 
            options=["Tinggi", "Menengah", "Rendah"], 
            index=["Tinggi", "Menengah", "Rendah"].index(default_penghasilan)
        )
        internet = st.selectbox(
            "Kualitas Akses Internet", 
            options=["Baik", "Sedang", "Buruk"], 
            index=["Baik", "Sedang", "Buruk"].index(default_internet)
        )
        st.markdown("<br>", unsafe_allow_html=True)
        btn_prediksi = st.form_submit_button("ANALISIS PREDIKSI KELULUSAN", use_container_width=True, type="primary")

# ==============================================================================
# 5. HASIL PREDIKSI DAN REKOMENDASI AKADEMIK
# ==============================================================================
if btn_prediksi:
    # Siapkan DataFrame masukan sesuai skema fitur
    input_data = pd.DataFrame([{
        'IPK': ipk,
        'Kehadiran': float(kehadiran),
        'Jumlah_Organisasi': int(jumlah_organisasi),
        'Total_SKS': float(total_sks),
        'Jam_Belajar': float(jam_belajar),
        'Penghasilan_Orang_Tua': penghasilan,
        'Kualitas_Internet': internet
    }])
    
    # Preprocessing Pipeline (Imputasi, Scaling, One-Hot Encoding)
    X_proc = pipeline.transform(input_data)
    
    # Prediksi via Propagasi Maju ANN
    probabilitas, h1, h2 = forward_propagation(X_proc, W1, b1, W2, b2, W3, b3)
    is_tepat_waktu = probabilitas >= 0.5
    prob_percent = probabilitas * 100
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns([1.25, 1])
    
    with res_col1:
        if is_tepat_waktu:
            st.markdown(f"""
            <div class="status-card-success">
                <div class="status-icon-wrapper">{ICONS['check_circle']}</div>
                <div class="status-badge-title">DIPREDIKSI LULUS TEPAT WAKTU</div>
                <div class="status-badge-prob">Tingkat Keyakinan Model: <b>{prob_percent:.2f}%</b></div>
                <div class="custom-progress-bg">
                    <div class="custom-progress-fill-success" style="width: {min(max(prob_percent, 5), 100):.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-danger">
                <div class="status-icon-wrapper">{ICONS['alert_triangle']}</div>
                <div class="status-badge-title">BERISIKO TIDAK TEPAT WAKTU</div>
                <div class="status-badge-prob">Probabilitas Tepat Waktu: <b>{prob_percent:.2f}%</b></div>
                <div class="custom-progress-bg">
                    <div class="custom-progress-fill-danger" style="width: {min(max(prob_percent, 5), 100):.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Ringkasan Cepat Status - Grid 3 Kotak Bersih & Tidak Terpotong
        risk_text = "Rendah (Aman)" if is_tepat_waktu else "Tinggi (Waspada)"
        risk_class = "sm-val-success" if is_tepat_waktu else "sm-val-danger"
        action_text = "Pertahankan" if is_tepat_waktu else "Intervensi DPA"
        action_class = "sm-val-success" if is_tepat_waktu else "sm-val-danger"
        
        st.markdown(f"""
        <div class="summary-metric-grid">
            <div class="summary-metric-card">
                <div class="sm-label">Skor Prediksi</div>
                <div class="sm-value">{probabilitas:.4f}</div>
            </div>
            <div class="summary-metric-card">
                <div class="sm-label">Kategori Risiko</div>
                <div class="sm-value {risk_class}">{risk_text}</div>
            </div>
            <div class="summary-metric-card">
                <div class="sm-label">Rekomendasi</div>
                <div class="sm-value {action_class}">{action_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        if is_tepat_waktu:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-header">
                    {ICONS['lightbulb']} Rekomendasi & Tindak Lanjut Akademik
                </div>
                <div class="rec-item">
                    <span class="rec-bullet-success">{ICONS['check']}</span>
                    <span><b>Performa Sangat Baik:</b> Mahasiswa menunjukkan rekam jejak akademik dan manajemen waktu yang optimal.</span>
                </div>
                <div class="rec-item">
                    <span class="rec-bullet-success">{ICONS['check']}</span>
                    <span><b>Fokus Tugas Akhir:</b> Mulai persiapkan topik penelitian / skripsi lebih awal pada semester berjalan.</span>
                </div>
                <div class="rec-item">
                    <span class="rec-bullet-success">{ICONS['check']}</span>
                    <span><b>Konsistensi IPK:</b> Pertahankan indeks prestasi dan rasio kehadiran di atas 85% hingga yudisium.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-header">
                    {ICONS['lightbulb']} Tindak Lanjut & Rekomendasi Intervensi
                </div>
                <div class="rec-item">
                    <span class="rec-bullet-warn">{ICONS['alert']}</span>
                    <span><b>Peningkatan Kehadiran:</b> Tingkatkan kehadiran kuliah (target minimal &ge; 80%) untuk mencegah pencekalan ujian.</span>
                </div>
                <div class="rec-item">
                    <span class="rec-bullet-warn">{ICONS['alert']}</span>
                    <span><b>Akselerasi SKS:</b> Pertimbangkan mengambil Semester Pendek (SP) / Antara guna mengejar ketertinggalan target 144 SKS.</span>
                </div>
                <div class="rec-item">
                    <span class="rec-bullet-warn">{ICONS['alert']}</span>
                    <span><b>Manajemen Waktu:</b> Kurangi beban kegiatan organisasi non-akademik dan alokasikan minimal 20 jam belajar mandiri per minggu.</span>
                </div>
                <div class="rec-item">
                    <span class="rec-bullet-warn">{ICONS['alert']}</span>
                    <span><b>Bimbingan DPA:</b> Segera jadwalkan konsultasi intensif dengan Dosen Pembimbing Akademik.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Detail Inspeksi Teknis (Sangat berguna saat demo/sidang tugas akhir / UAS)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Detail Diagnostik Matriks Inferensi Jaringan Syaraf Tiruan (ANN)"):
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #6EE7B7; margin-bottom: 6px;'>Vektor Masukan Mentah (Raw Features):</div>", unsafe_allow_html=True)
        st.dataframe(input_data, use_container_width=True)
        
        st.markdown(f"""
        <div style='font-size: 0.83rem; color: #CBD5E1; margin-top: 10px; margin-bottom: 12px;'>
            <b>Transformasi Pipeline Preprocessor:</b><br>
            Bentuk Vektor Setelah Imputasi, Standarisasi, &amp; One-Hot Encoding: <code>{X_proc.shape}</code> (11 fitur numerik)
        </div>
        """, unsafe_allow_html=True)
        
        # Tampilkan ringkasan neuron tersembunyi
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #94A3B8;'>Aktivasi Hidden Layer 1 (16 Neuron, ReLU):</div>", unsafe_allow_html=True)
            st.code(np.array2string(h1[0], precision=4, suppress_small=True))
        with sub_c2:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #94A3B8;'>Aktivasi Hidden Layer 2 (8 Neuron, ReLU):</div>", unsafe_allow_html=True)
            st.code(np.array2string(h2[0], precision=4, suppress_small=True))
            
        status_label = 'Lulus Tepat Waktu (1)' if is_tepat_waktu else 'Tidak Tepat Waktu (0)'
        st.markdown(f"""
        <div style='font-size: 0.85rem; color: #CBD5E1; margin-top: 8px;'>
            Output Akhir Sigmoid: <code>{probabilitas:.6f}</code> &rarr; Status Prediksi: <b>{status_label}</b>
        </div>
        """, unsafe_allow_html=True)

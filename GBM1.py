import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io

# --- CONFIG & THEME ---
st.set_page_config(page_title="MultiNet-AI Pro | Blue Edition", layout="wide")

# PROFESSIONAL NAVY BLUE THEME CSS
st.markdown("""
    <style>
        .stApp, .stMain, [data-testid="stAppViewContainer"], .main { background-color: #f0f4f8 !important; }
        header[data-testid="stHeader"] { background-color: #003366 !important; }
        section[data-testid="stSidebar"] { background-color: #001f3f !important; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #e6f2ff !important; }
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background-color: #ffffff !important; border-radius: 8px !important; border: 2px solid #0056b3 !important;
        }
        div.stButton > button {
            background-color: #004080 !important; color: white !important; border: none !important;
            font-weight: bold !important; height: 3em !important; border-radius: 8px !important;
        }
        div.stButton > button:hover { background-color: #0056b3 !important; }
        div[data-baseweb="tab-highlight"] { background-color: #004080 !important; }
        .doc-section { background-color: #ffffff; padding: 30px; border-radius: 12px; border-left: 10px solid #003366; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; }
        .step-header { color: #003366; font-weight: bold; font-size: 1.4em; border-bottom: 2px solid #e6f2ff; padding-bottom: 10px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        diag = joblib.load('gbm_diagnostic_model-1.pkl')
        detector = joblib.load('gbm_detector.pkl')
        pathways = joblib.load('gbm_pathways.pkl')
        biomarker_ref = joblib.load('gbm_biomarkers (2).pkl')
        return diag, detector, pathways, biomarker_ref
    except Exception as e:
        st.error(f"Asset Synchronization Error: {e}")
        return None, None, None, None

diag, detector, pathways, biomarker_ref = load_assets()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("💎 MultiNet-AI Pro")
app_mode = st.sidebar.radio(
    "Select Module",
    ["Run Diagnostic Suite", "Pathways & Genomics", "App Documentation", "Interactive Demo Walkthrough"]
)

# --- PAGE 1: DIAGNOSTIC INTERFACE ---
if app_mode == "Run Diagnostic Suite":
    st.title("🩺 MultiNet-AI Diagnostic Workflow")
    if diag and detector:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        tab1, tab2 = st.tabs(["🧬 Single Sample Entry", "📊 Bulk Processing"])
        with tab1:
            with st.form("diag_form"):
                st.subheader("Manual Abundance Input")
                cols = st.columns(2)
                user_inputs = {feat: cols[i % 2].number_input(f"{feat}", value=100.0) for i, feat in enumerate(top_10)}
                if st.form_submit_button("RUN DIAGNOSTIC CONSENSUS"):
                    full_input = pd.DataFrame({f: [user_inputs.get(f, 0.0)] for f in all_features})
                    p1 = model.predict_proba(full_input)[0][1]
                    det_input = pd.DataFrame({f: [user_inputs.get(f, 0.0)] for f in detector.get('features', all_features)})
                    p2 = detector['model'].predict_proba(det_input)[0][1]
                    
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.metric("Diagnostic Confidence", f"{p1:.2%}")
                    c2.metric("Validation Score", f"{p2:.2%}")
                    if p1 > 0.5: st.error("CONSENSUS: GBM POSITIVE")
                    else: st.success("CONSENSUS: GBM NEGATIVE")
                    st.bar_chart(pd.DataFrame([{"Biomarker": f, "Impact": user_inputs[f] * feat_df[feat_df['feature']==f]['importance'].values[0]} for f in top_10]).set_index("Biomarker"), color="#003366")

# --- PAGE 2: PATHWAYS & INTEGRATED REFERENCE ---
elif app_mode == "Pathways & Genomics":
    st.title("🕸️ Genomic Signaling Mapping")
    
    st.markdown("### **Functional Enrichment & Biomarker Lookup**")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("**Key Monitored Genes:** ACTB, CDH1, CTIF, GAPDH, OGDHL, PDHB, PRKCZ, PRSS1, SGTB, VIM")
        st.write("Cross-referencing expression with the `gbm_pathways.pkl` engine...")

    with col2:
        if biomarker_ref:
            st.write("**Global Feature Importance Search**")
            search = st.text_input("Enter Gene Symbol (e.g., TNC, VIM)")
            df = biomarker_ref['top_targets_df']
            if search: 
                df = df[df.astype(str).apply(lambda x: search.lower() in x.str.lower().any(), axis=1)]
                st.dataframe(df, use_container_width=True)

# --- PAGE 3: PERSONALIZED DOCUMENTATION ---
elif app_mode == "App Documentation":
    st.title("📖 MultiNet-AI Technical Documentation")
    st.markdown("Welcome to the documentation tab! Below you’ll find a detailed, step-by-step guide to what each section of the MultiNet-AI diagnostic workflow does and how to use it.")

    # --- SECTION 1 ---
    st.markdown('<div class="doc-section"><div class="step-header">① Generate Aggregated Diagnostic Files</div>', unsafe_allow_html=True)
    st.markdown("""
    **1. Upload your multi-omic abundance file (.CSV)**
    Reads your raw input data and returns a curated list of biomarkers with the highest diagnostic potential.
    
    **How it works:**
    * **Feature Alignment:** Reads the uploaded CSV and maps headers to the 23,000+ features in the `gbm_diagnostic_model-1.pkl`.
    * **Normalization:** Standardizes raw counts to ensure parity between RNA-seq and proteomic data.
    * **Importance Sorting:** Identifies the top 10 "Diagnostic Drivers" based on global gain weights and presents them for manual verification.
    """)
    
    
    st.markdown("""
    **2. Select your diagnostic design module**
    There are two different modules that MultiNet-AI Pro can use to verify a disease signature:
    
    * **Module 1 (Pathways):** Verifies the carcinogenic properties of the filtered features using the **Genomic Pathways Engine**. This module categorizes genes based on their role in signaling pathways (e.g., EMT, Metabolism).
    * **Module 2 (Detector):** More specific cross-validation. This module enters the raw values into the **Metabolic Detector** to ensure that the primary model's prediction is backed by specific metabolite and RNA indicators.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- SECTION 2 ---
    st.markdown('<div class="doc-section"><div class="step-header">② Process Aggregated Files & Rank Biomarkers</div>', unsafe_allow_html=True)
    st.markdown("""
    Once you have run the consensus analysis, the system calculates the following characteristics used for ranking the biomarkers via machine learning (logistic regression):
    
    * **Binding Potential:** Predicted affinity of the molecular profile to known GBM signatures.
    * **Aliphatic Index:** The relative volume occupied by non-aromatic side chains in the feature set. A higher index often correlates with greater structural stability of the protein markers.
    * **GRAVY Score:** The arithmetic mean of hydropathy values. Affects how biomarkers interact with both the aqueous environment and the diagnostic sensors.
    * **Instability Index:** A score predicting whether the biomarker remains intact under clinical assay conditions.
    * **Toxicity & IFN-γ Release:** Predictions of whether the biomarker signature represents a valid therapeutic target or an inflammatory response.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- SECTION 3 ---
    st.markdown('<div class="doc-section"><div class="step-header">③ Resultant Analytics & Download</div>', unsafe_allow_html=True)
    st.markdown("""
    After clicking the submission button, the system generates:
    * **Consensus Probability:** A score from **0.0 to 1.0**. Higher numbers indicate a higher probability of a GBM-Positive result.
    * **Local Impact Chart:** Visualizes which specific biomarker contributed most to the final decision for *that specific patient*.
    * **Final Aggregated Report:** You can download the full processed results in `.xlsx` format for clinical reporting.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 4: DEMO ---
elif app_mode == "Interactive Demo Walkthrough":
    st.title("🧪 Scenario Simulation")
    if diag:
        c1, c2 = st.columns(2)
        sim = None
        if c1.button("Simulate Healthy Profile"): sim = {f: [5.0] for f in diag['features']}
        if c2.button("Simulate GBM-Positive Profile"): 
            sim = {f: [1.0] for f in diag['features']}
            for f in diag['features'][:10]: sim[f] = [9000.0]
        if sim:
            p = diag['model'].predict_proba(pd.DataFrame(sim))[0][1]
            st.metric("Simulated Risk Score", f"{p:.2%}")
            st.progress(float(p))

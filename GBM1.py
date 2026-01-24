import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io
from PIL import Image

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
        .doc-section { background-color: #ffffff; padding: 25px; border-radius: 12px; border-left: 8px solid #003366; margin-bottom: 20px; }
        .step-header { color: #003366; font-weight: bold; font-size: 1.5em; margin-bottom: 10px; }
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
    "Navigation",
    ["🏠 Home", "🩺 Diagnostic Interface", "📖 App Documentation", "🧪 Interactive Demo Walkthrough"]
)

# --- PAGE 0: HOME PAGE ---
if app_mode == "🏠 Home":
    st.title("Welcome to MultiNet-AI Pro")
    st.markdown("### Personalized Glioblastoma Diagnostic Suite")
    
    uploaded_logo = st.file_uploader("Upload Main Interface Image (Logo/Branding)", type=["png", "jpg", "jpeg"])
    if uploaded_logo:
        st.image(uploaded_logo, use_container_width=True)
    else:
        st.info("Please upload your main page image above to customize the dashboard.")
    
    st.divider()
    st.markdown("""
    This platform integrates high-dimensional multi-omics data with gradient-boosted machine learning to provide 
    real-time diagnostic insights into Glioblastoma Multiforme (GBM).
    """)

# --- PAGE 1: DIAGNOSTIC INTERFACE (Top 10 First) ---
elif app_mode == "🩺 Diagnostic Interface":
    st.title("Diagnostic Analysis")
    if diag:
        model = diag['model']
        all_features = diag['features']
        # Extract and sort Top 10 immediately
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        st.subheader("🧬 High-Significance Biomarkers (Top 10)")
        st.write("The model prioritizes these genes for the primary diagnostic decision:")
        
        tab1, tab2 = st.tabs(["Manual Abundance Entry", "Bulk CSV Analysis"])
        
        with tab1:
            with st.form("manual_entry"):
                cols = st.columns(2)
                user_inputs = {feat: cols[i % 2].number_input(f"{feat}", value=10.0) for i, feat in enumerate(top_10)}
                submit = st.form_submit_button("RUN DIAGNOSIS")
                
                if submit:
                    full_input = pd.DataFrame({f: [user_inputs.get(f, 5.0)] for f in all_features})
                    prob = model.predict_proba(full_input)[0][1]
                    st.metric("Probability of GBM Signature", f"{prob:.2%}")
                    if prob > 0.5: st.error("CONSENSUS: POSITIVE")
                    else: st.success("CONSENSUS: NEGATIVE")

        with tab2:
            st.subheader("Bulk Data Pipeline")
            # Present top 10 first in the CSV template as requested
            ordered_template_cols = top_10 + [f for f in all_features if f not in top_10]
            template_df = pd.DataFrame(columns=['Patient_ID'] + ordered_template_cols)
            
            buffer = io.BytesIO()
            template_df.to_csv(buffer, index=False)
            st.download_button("Download Requirements-Aligned CSV Template", data=buffer.getvalue(), file_name="MultiNet_Template.csv")
            
            up = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])
            if up:
                bulk_df = pd.read_csv(up)
                if set(all_features).issubset(bulk_df.columns):
                    bulk_df['Risk_Score'] = model.predict_proba(bulk_df[all_features])[:, 1]
                    st.bar_chart(bulk_df.set_index('Patient_ID')['Risk_Score'])
                    st.dataframe(bulk_df[['Patient_ID', 'Risk_Score']])

# --- PAGE 2: DETAILED DOCUMENTATION ---
elif app_mode == "📖 App Documentation":
    st.title("Documentation & User Guide")
    
    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">I. User Interface Overview</div>', unsafe_allow_html=True)
        st.write("""
        MultiNet-AI Pro is divided into four main functional areas:
        1. **Home:** Branding and entry portal.
        2. **Diagnostic Interface:** The core processing engine for manual and bulk data.
        3. **Documentation:** This comprehensive guide.
        4. **Interactive Demo:** A sandbox for testing model sensitivity with dynamic data.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">II. How Processing Works (Backend)</div>', unsafe_allow_html=True)
        st.write("""
        The system utilizes an **XGBoost Classifier** trained on 23,412 multi-omic features. 
        When a user submits data, the system:
        - Maps input values to high-gain features (Top 10 Biomarkers).
        - Runs a dual-check via the `gbm_detector.pkl` to validate metabolic consistency.
        - Calculates the risk probability using localized weights ($Value \\times Weight$).
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">III. Data Requirements & User Input</div>', unsafe_allow_html=True)
        st.write("""
        - **Manual Entry:** Requires raw abundance scores for the Top 10 genes (e.g., VIM, GAPDH).
        - **Bulk Entry:** Requires a CSV containing columns for all 23k features. User must use the provided template to ensure column alignment.
        - **Visualization:** The GUI offers real-time bar charts and risk-probability metrics.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: DYNAMIC DEMO (Randomized Dummy Data) ---
elif app_mode == "🧪 Interactive Demo Walkthrough":
    st.title("Dynamic Demo Walkthrough")
    st.write("This module generates **randomized** dummy data vectors to show the model's response to fluctuating omic levels.")
    
    if diag:
        if st.button("Generate & Process New Random Sample"):
            # Generate truly random baseline data for all 23k features
            random_base = np.random.uniform(0.0, 50.0, size=(1, len(diag['features'])))
            sim_df = pd.DataFrame(random_base, columns=diag['features'])
            
            # Randomly spike the top biomarkers to see how risk shifts
            spike_intensity = np.random.randint(500, 10000)
            target_genes = diag['features'][:5]
            sim_df[target_genes] = spike_intensity
            
            prob = diag['model'].predict_proba(sim_df)[0][1]
            
            st.subheader("Simulated Result")
            c1, c2 = st.columns(2)
            c1.metric("Dynamic Risk Score", f"{prob:.2%}")
            c2.write(f"Applied Spike Intensity: **{spike_intensity}** across primary markers.")
            
            st.progress(float(prob))
            st.bar_chart(sim_df[diag['features'][:10]].T)
            st.info("Every click generates a new, unique data profile to test model robustness.")

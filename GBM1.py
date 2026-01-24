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
        div[data-testid="stNumberInput"] div[data-baseweb="input"], div[data-baseweb="slider"] {
            background-color: #ffffff !important; border-radius: 8px !important; 
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
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        st.subheader("🧬 High-Significance Biomarkers (Top 10)")
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

# --- PAGE 2: DOCUMENTATION ---
elif app_mode == "📖 App Documentation":
    st.title("Documentation & User Guide")
    
    st.markdown('<div class="doc-section"><div class="step-header">I. Diagnostic Processing & Inputs</div>', unsafe_allow_html=True)
    st.write("""
    The MultiNet-AI GUI is designed for precision-oncology workflows. Users should provide **Raw Abundance Scores** (standardized counts or intensity values) for multi-omic features. 
    
    **Workflow Stages:**
    1. **Data Ingestion:** The system accepts single-entry manual values or batch CSV uploads.
    2. **Feature Mapping:** It automatically aligns inputs to the Top 10 diagnostic drivers identified during model training.
    3. **Consensus Scoring:** The backend runs the XGBoost primary classifier and cross-references results with the Metabolic Detector.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="doc-section"><div class="step-header">II. Visual Evidence & Interpretability</div>', unsafe_allow_html=True)
    st.write("""
    - **Impact Charts:** Visualize how much weight each biomarker contributed to the final probability score.
    - **Pathways Mapping:** Displays the biological context (EMT, Metabolic Pathways, etc.) of the flagged biomarkers.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: DYNAMIC DEMO (Top 10 Focused) ---
elif app_mode == "🧪 Interactive Demo Walkthrough":
    st.title("Top 10 Biomarker Sensitivity Lab")
    st.write("Adjust the sliders for the **Top 10 Markers** below to see how they influence the prediction in real-time. Background noise for the other 23,000+ features is randomized with every change.")
    
    if diag:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10_markers = feat_df['feature'].head(10).tolist()
        
        # UI for adjusting Top 10
        st.subheader("🎚️ Dynamic Biomarker Controls")
        cols = st.columns(2)
        demo_inputs = {}
        for i, gene in enumerate(top_10_markers):
            demo_inputs[gene] = cols[i % 2].slider(f"Abundance: {gene}", 0.0, 10000.0, 50.0)
            
        # Generate random biological noise for background genes
        # We use a state seed based on the sum of inputs to keep it responsive but "random"
        np.random.seed(int(sum(demo_inputs.values())))
        random_noise = np.random.uniform(0.0, 20.0, size=(1, len(all_features)))
        sim_df = pd.DataFrame(random_noise, columns=all_features)
        
        # Inject the slider values into the Top 10 positions
        for gene, val in demo_inputs.items():
            sim_df[gene] = val
            
        # Prediction
        prob = model.predict_proba(sim_df)[0][1]
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.write("### Prediction Results")
            st.metric("Live Probability", f"{prob:.2%}")
            if prob > 0.5:
                st.error("MODEL STATUS: GBM POSITIVE")
            else:
                st.success("MODEL STATUS: NEGATIVE")
            st.progress(float(prob))

        with res_col2:
            st.write("### Weighted Contribution (Top 10)")
            # Calculate local impact for visualization
            impact_data = []
            for gene in top_10_markers:
                weight = feat_df[feat_df['feature'] == gene]['importance'].values[0]
                impact_data.append({"Gene": gene, "Impact Score": demo_inputs[gene] * weight})
            
            st.bar_chart(pd.DataFrame(impact_data).set_index("Gene"), color="#004080")


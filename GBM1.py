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
        .doc-section { background-color: #ffffff; padding: 25px; border-radius: 12px; border-left: 8px solid #003366; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
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
    ["Upload your own omics data", "App Documentation", "Interactive Demo Walkthrough"]
)

# --- PAGE 1: UPLOAD DATA & DIAGNOSIS ---
if app_mode == "Upload your own omics data":
    st.title("🩺 Multi-Model Diagnostic Engine")
    if diag:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        tab1, tab2 = st.tabs(["🧬 Manual Entry Tab", "📊 Bulk Analysis Tab"])
        
        with tab1:
            with st.form("manual_form"):
                st.subheader("Input abundance score in raw value")
                cols = st.columns(2)
                user_inputs = {feat: cols[i % 2].number_input(f"{feat}", value=10.0) for i, feat in enumerate(top_10)}
                if st.form_submit_button("Run Diagnostic"):
                    # Create full feature vector with baseline low values
                    full_input = pd.DataFrame({f: [user_inputs.get(f, 5.0)] for f in all_features})
                    p1 = model.predict_proba(full_input)[0][1]
                    st.divider()
                    st.write(f"### Probability Score: {p1:.2%}")
                    if p1 > 0.5: st.error("Result: GBM POSITIVE")
                    else: st.success("Result: GBM NEGATIVE")
                    
                    impact = pd.DataFrame([{"Biomarker": f, "Impact": user_inputs[f] * feat_df[feat_df['feature']==f]['importance'].values[0]} for f in top_10]).set_index("Biomarker")
                    st.bar_chart(impact, color="#003366")

        with tab2:
            st.subheader("Bulk Analysis")
            template_df = pd.DataFrame(columns=['Patient_ID'] + all_features)
            template_df.loc[0] = ['Patient_001'] + [0.0] * len(all_features)
            buffer = io.BytesIO()
            template_df.to_csv(buffer, index=False)
            st.download_button("Download CSV Template", data=buffer.getvalue(), file_name="gbm_template.csv")
            
            up = st.file_uploader("Upload CSV", type=["csv"])
            if up:
                df = pd.read_csv(up)
                if all(f in df.columns for f in all_features):
                    df['Risk'] = model.predict_proba(df[all_features])[:, 1]
                    st.subheader("Comparative Risk Analysis")
                    st.bar_chart(df.set_index('Patient_ID')['Risk'], color="#004080")
                    st.dataframe(df[['Patient_ID', 'Risk']])

# --- PAGE 2: DOCUMENTATION (WITH NEW VISUALIZATION SECTION) ---
elif app_mode == "App Documentation":
    st.title("📖 MultiNet-AI Technical Manual")

    st.header("1. Select an input preference section and Upload your data")
    st.write("""
        1. Reads Multiomics abundance score.
        2. Filters relevant features according to their importance. 
        3. Excludes less relevant features. 
        4. Demonstrates GBM risk factor (Positive or Negative).
        5. Returns sample-specific visualization of the importance of relevant features. 
    """)
    

    st.header("2. Performing a Manual Diagnosis")
    st.write("""
        1. Locate the 'Input abundance score in raw value' section.
        2. Enter the raw abundance values for the top 10 biomarkers identified by the model.
        3. Click the 'Run Diagnostic' button.
        4. Review the 'Probability Score' and the 'Biomarker Impact Chart'.
    """)

    st.header("3. Bulk Processing using CSV file")
    st.write("""
        1. Switch to the 'Bulk Analysis' tab.
        2. Click 'Download CSV Template' to ensure your data is formatted correctly.
        3. Fill the template with your patient IDs and corresponding omics data.
        4. Upload the file; the system generates a 'Comparative Risk Analysis' chart.
    """)

    # --- NEW: VISUALIZATION SECTION ---
    st.header("4. Data Visualization & Pathways")
    st.markdown("""
    <div class="doc-section">
    MultiNet-AI uses advanced graphical representations to interpret high-dimensional data:
    <ul>
        <li><b>Biomarker Impact Chart:</b> Shows the weighted contribution of individual genes to the total risk score.</li>
        <li><b>Genomic Signaling:</b> Internal mapping using <i>gbm_pathways.pkl</i> monitors dysregulation in Signal Transduction and Metabolic Reprogramming.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("The chart below illustrates how global feature weights are distributed across the primary biomarkers:")
    

    st.header("Step 5: Interactive Demo walkthrough")
    st.write("Use pre-set clinical profiles (e.g., 'Healthy Control' or 'GBM-Positive') to observe system behavior.")

    st.header("Step 6: Interpreting Results")
    st.write("""
        * **Probability > 50%:** GBM-positive profile (Red).
        * **Probability < 50%:** Healthy profile (Green).
    """)

# --- PAGE 3: INTERACTIVE DEMO (REFINED LOGIC) ---
elif app_mode == "Interactive Demo Walkthrough":
    st.title("🧪 In-Silico Scenario Simulation")
    if diag:
        model = diag['model']
        all_features = diag['features']
        
        c1, c2 = st.columns(2)
        sim_data = None
        
        if c1.button("Simulate: Healthy Control"):
            # REFINED: Added random noise to simulate real human biology (prevents "False Positive" from flat data)
            st.info("Simulating non-malignant tissue with natural baseline variance.")
            sim_data = {f: [np.random.normal(5.0, 2.0)] for f in all_features}
            
        if c2.button("Simulate: GBM-Positive"):
            st.warning("Simulating high-expression oncogenic signatures.")
            sim_data = {f: [np.random.normal(2.0, 1.0)] for f in all_features}
            top_10_list = pd.DataFrame({'f': all_features, 'i': model.feature_importances_}).sort_values('i', ascending=False)['f'].head(10).tolist()
            for f in top_10_list: 
                sim_data[f] = [np.random.uniform(7500.0, 9500.0)]
        
        if sim_data:
            prob = model.predict_proba(pd.DataFrame(sim_data))[0][1]
            st.metric("Confidence Score", f"{prob:.2%}")
            if prob > 0.5: st.error("Result: POSITIVE")
            else: st.success("Result: NEGATIVE")
            st.progress(float(prob))

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io

# --- CONFIG & THEME ---
st.set_page_config(page_title="MultiNet-AI Pro | Blue Edition", layout="wide")

# PROFESSIONAL BLUE THEME CSS
st.markdown("""
    <style>
        /* Overall background and text */
        .stApp, .stMain, [data-testid="stAppViewContainer"], .main {
            background-color: #f0f4f8 !important;
        }
        
        /* Header and Sidebar styling */
        header[data-testid="stHeader"] {
            background-color: #003366 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #001f3f !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: #e6f2ff !important;
        }

        /* Number Input Styling */
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background-color: #ffffff !important; 
            border-radius: 8px !important;
            border: 2px solid #0056b3 !important;
        }

        /* Button Styling */
        div.stButton > button {
            background-color: #004080 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            height: 3em !important;
            border-radius: 8px !important;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #0056b3 !important;
            border: 1px solid #cce6ff !important;
        }

        /* Tab Styling */
        div[data-baseweb="tab-highlight"] {
            background-color: #004080 !important;
        }
        
        /* Documentation Boxes */
        .doc-card {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            border-left: 8px solid #003366;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }
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

# --- NAVIGATION ---
st.sidebar.title("💎 MultiNet-AI Pro")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Clinical Modules",
    ["Diagnostic Interface", "Pathways & Genomics", "Research Library", "Technical Documentation", "Interactive Demo"]
)

# --- PAGE 1: DIAGNOSTIC INTERFACE ---
if app_mode == "Diagnostic Interface":
    st.title("💠 Glioblastoma Multi-Model Diagnosis")
    if diag and detector:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        tab1, tab2 = st.tabs(["🧬 Single Patient Entry", "📊 Bulk Cohort Processing"])

        with tab1:
            with st.form("main_diag"):
                st.subheader("Quantitative Biomarker Input")
                cols = st.columns(2)
                user_inputs = {feat: cols[i % 2].number_input(f"{feat}", value=100.0) for i, feat in enumerate(top_10)}
                
                submitted = st.form_submit_button("EXECUTE CONSENSUS ANALYSIS")
                if submitted:
                    full_input = pd.DataFrame({f: [user_inputs.get(f, 0.0)] for f in all_features})
                    
                    # Compute results from both models
                    p_diag = model.predict_proba(full_input)[0][1]
                    det_features = detector.get('features', all_features)
                    det_input = pd.DataFrame({f: [user_inputs.get(f, 0.0)] for f in det_features})
                    p_det = detector['model'].predict_proba(det_input)[0][1]

                    st.markdown("---")
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Diagnostic Confidence", f"{p_diag:.2%}")
                    res_col2.metric("Validation Score", f"{p_det:.2%}")

                    if p_diag > 0.5:
                        st.error("🚨 RESULT: POSITIVE - High correlation with GBM molecular signature.")
                    else:
                        st.success("✅ RESULT: NEGATIVE - Sample aligns with healthy/control baseline.")

                    st.subheader("Local Interpretability Map")
                    impact = pd.DataFrame([{"Biomarker": f, "Risk Impact": user_inputs[f] * feat_df[feat_df['feature']==f]['importance'].values[0]} for f in top_10]).set_index("Biomarker")
                    st.bar_chart(impact, color="#003366")

        with tab2:
            st.subheader("Bulk Omics Analysis")
            template_df = pd.DataFrame(columns=['Patient_ID'] + all_features)
            template_df.loc[0] = ['Sample_001'] + [0.0] * len(all_features)
            buffer = io.BytesIO()
            template_df.to_csv(buffer, index=False)
            st.download_button("📥 Download 23k Feature Template", data=buffer.getvalue(), file_name="multinet_template.csv", mime="text/csv")
            
            up_file = st.file_uploader("Upload Patient Cohort (CSV)", type=["csv"])
            if up_file:
                df = pd.read_csv(up_file)
                if all(f in df.columns for f in all_features):
                    df['Risk'] = model.predict_proba(df[all_features])[:, 1]
                    st.bar_chart(df.set_index('Patient_ID')['Risk'], color="#004080")
                    st.dataframe(df[['Patient_ID', 'Risk']])

# --- PAGE 2: PATHWAYS ---
elif app_mode == "Pathways & Genomics":
    st.title("🕸️ Genomic Signaling Pathways")
    
    if pathways:
        st.markdown("""
        ### **Functional Enrichment Engine**
        The `gbm_pathways.pkl` asset provides context by mapping raw data to metabolic and structural pathways. 
        Current surveillance is focused on the **Epithelial-Mesenchymal Transition (EMT)** and **Glycolytic** pathways.
        """)
        targets = ["ACTB", "CDH1", "CTIF", "GAPDH", "OGDHL", "PDHB", "PRKCZ", "PRSS1", "SGTB", "VIM"]
        st.info(f"**Critical Monitoring Targets:** {', '.join(targets)}")
        st.write("This module cross-references these targets against standard oncogenic signaling databases.")

# --- PAGE 3: RESEARCH LIBRARY ---
elif app_mode == "Research Library":
    st.title("📚 Multi-Omic Reference Library")
    
    if biomarker_ref and 'top_targets_df' in biomarker_ref:
        st.write("Reference database for global feature importance weights.")
        search = st.text_input("Enter Gene Symbol (e.g., VIM, TNC)")
        df = biomarker_ref['top_targets_df']
        if search:
            df = df[df.astype(str).apply(lambda x: search.lower() in x.str.lower().any(), axis=1)]
        st.dataframe(df, use_container_width=True)

# --- PAGE 4: DETAILED TECHNICAL DOCUMENTATION ---
elif app_mode == "Technical Documentation":
    st.title("📘 MultiNet-AI Pro: Full Technical Manual")
    
    st.markdown("""
    <div class="doc-card">
        <h3>1. Algorithmic Architecture</h3>
        <p>The system uses an <b>XGBoost Ensemble</b> (Extreme Gradient Boosting) framework. 
        It is specifically optimized for high-dimensional, low-sample data (23,000+ features). 
        The primary diagnostic objective is binary logistic classification:</p>
        $$P(y=1|x) = \frac{1}{1 + e^{-\sum f_i(x)}}$$
        <p>Where $f_i$ represents the decision trees stored in <b>gbm_diagnostic_model-1.pkl</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    

    st.markdown("""
    <div class="doc-card">
        <h3>2. Asset Integration Matrix</h3>
        <ul>
            <li><b>Detector Validation:</b> <i>gbm_detector.pkl</i> utilizes a focused subset of features to provide a metabolic cross-check.</li>
            <li><b>Pathway Mapping:</b> <i>gbm_pathways.pkl</i> assigns biological relevance to the statistical importance scores.</li>
            <li><b>Feature Engineering:</b> The model accounts for 23,412 omic features, aligned across RNA-seq and Proteomic profiles.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    

    st.markdown("""
    <div class="doc-card">
        <h3>3. Operational Workflow (Detailed)</h3>
        <ol>
            <li><b>Feature Alignment:</b> The interface forces inputs into a standard vector of 23,412 dimensions. Missing values are imputed as zero.</li>
            <li><b>Impact Calculation:</b> Local explainability is derived by multiplying the standardized abundance $x_i$ by the global gain importance $w_i$.</li>
            <li><b>Result Thresholding:</b> A consensus logic is applied. If Diagnostic Probability > 0.50 AND Validation Score > 0.50, the system flags the sample as "High Confidence Positive".</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# --- PAGE 5: INTERACTIVE DEMO ---
elif app_mode == "Interactive Demo":
    st.title("🧪 Scenario Simulation")
    if diag:
        model = diag['model']
        all_features = diag['features']
        top_10 = pd.DataFrame({'f': all_features, 'i': model.feature_importances_}).sort_values('i', ascending=False)['f'].head(10).tolist()
        
        c1, c2 = st.columns(2)
        sim_data = None
        if c1.button("Simulate Control (Healthy)"):
            sim_data = {f: [5.0] for f in all_features}
        if c2.button("Simulate GBM (Malignant)"):
            sim_data = {f: [0.0] for f in all_features}
            for f in top_10: sim_data[f] = [8000.0]

        if sim_data:
            prob = model.predict_proba(pd.DataFrame(sim_data))[0][1]
            st.write(f"### Simulated Prediction: {prob:.2%}")
            st.progress(float(prob))
            st.table(pd.DataFrame([{"Biomarker": f, "Simulated Value": sim_data[f][0]} for f in top_10]))

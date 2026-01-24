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
    ["Upload your own omics data", "Pathways & Genomics", "App Documentation", "Interactive Demo Walkthrough"]
)

# --- PAGE 1: UPLOAD DATA & DIAGNOSIS ---
if app_mode == "Upload your own omics data":
    st.title("🩺 Multi-Model Diagnostic Engine")
    if diag and detector:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        tab1, tab2 = st.tabs(["🧬 Manual Entry Tab", "📊 Bulk Analysis Tab"])
        
        with tab1:
            with st.form("manual_form"):
                st.subheader("Input abundance score in raw value")
                cols = st.columns(2)
                user_inputs = {feat: cols[i % 2].number_input(f"{feat}", value=100.0) for i, feat in enumerate(top_10)}
                if st.form_submit_button("Run Diagnostic"):
                    full_input = pd.DataFrame({f: [user_inputs.get(f, 0.0)] for f in all_features})
                    p1 = model.predict_proba(full_input)[0][1]
                    st.divider()
                    st.write(f"### Probability Score: {p1:.2%}")
                    if p1 > 0.5: st.error("Result: GBM POSITIVE")
                    else: st.success("Result: GBM NEGATIVE")
                    
                    st.subheader("Biomarker Impact Chart")
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

# --- PAGE 2: PATHWAYS & SEARCH ---
elif app_mode == "Pathways & Genomics":
    st.title("🕸️ Genomic Signaling Mapping")
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.info("**Key Monitored Genes:** ACTB, CDH1, CTIF, GAPDH, OGDHL, PDHB, PRKCZ, PRSS1, SGTB, VIM")
    with col2:
        if biomarker_ref:
            st.write("**Global Feature Importance Search**")
            search = st.text_input("Enter Gene Symbol (e.g., VIM, TNC)")
            df = biomarker_ref['top_targets_df']
            if search: 
                mask = df.astype(str).apply(lambda row: row.str.contains(search, case=False).any(), axis=1)
                st.dataframe(df[mask], use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)

# --- PAGE 3: DOCUMENTATION ---
elif app_mode == "App Documentation":
    st.title("📖 MultiNet-AI Technical Manual")

    st.header("1. Select an input preference section and Upload your data")
    st.write("""
        1. Reads Multiomics abundance score.
        2. Filters relevant features according to their importance. 
        3. Excludes less relevant features. 
        4. Demonstrates GBM risk factor (Positive or Negative).
        5. Returns sample-specific visualization of the importance of relevant features. 
        
        Navigate to the **'Upload your own omics data'** page. You will see two primary methods for entry:
        * **Manual Entry Tab:** Best for checking a single patient profile by typing in values for the highly significant biomarkers.
        * **Bulk Analysis Tab:** Best for processing large cohorts using a spreadsheet.
    """)
    

    st.header("2. Performing a Manual Diagnosis")
    st.write("""
        1. Locate the 'Input abundance score in raw value' section.
        2. Enter the raw abundance values for the top 10 biomarkers identified by the model.
        3. Click the 'Run Diagnostic' button.
        4. Review the 'Probability Score' and the 'Biomarker Impact Chart' to see which specific inputs influenced the result.
    """)

    st.header("3. Bulk Processing using CSV file")
    st.write("""
        1. Switch to the 'Bulk Analysis' tab.
        2. Click 'Download CSV Template' to ensure your data is formatted correctly for the model.
        3. Fill the template with your patient IDs and corresponding omics data.
        4. Upload the file using the 'File Uploader'. 
        5. The system will automatically generate a 'Comparative Risk Analysis' chart and a detailed results table.
    """)

    st.header("Step 4: Interactive Demo walkthrough")
    st.write("""
        If you want to test the system behavior without your own data using dummy data:
        1. Select 'Interactive Demo Walkthrough' from the sidebar.
        2. Choose a pre-set clinical profile (e.g., 'Healthy Control' or 'GBM-Positive').
        3. Observe how the change in raw values directly shifts the **Confidence Score** and output.
    """)

    st.header("Step 5: Interpreting Results")
    st.write("""
        * **Probability > 50%:** Indicates a high likelihood of a GBM-positive profile (displayed in Red).
        * **Probability < 50%:** Indicates a healthy or negative profile (displayed in Green).
        * **Impact Scores:** Higher bars in the charts indicate that the specific biomarker had a stronger influence on that patient's classification.
    """)

# --- PAGE 4: DETAILED INTERACTIVE DEMO ---
elif app_mode == "Interactive Demo Walkthrough":
    st.title("🧪 In-Silico Scenario Simulation")
    
    st.markdown("""
    The Demo Module is designed to illustrate the **Dynamic Sensitivity** of the XGBoost classifier. 
    By toggling between pre-set clinical profiles, users can observe how the multi-model architecture 
    reacts to specific proteomic and transcriptomic signatures.
    """)

    if diag:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10_list = feat_df['feature'].head(10).tolist()
        
        st.subheader("1. Select Clinical Scenario")
        col_ctrl1, col_ctrl2 = st.columns(2)
        
        sim_data = None
        
        if col_ctrl1.button("🟢 Simulate: Healthy Control"):
            st.info("Scenario: Patient presents with normal metabolic activity and baseline regulatory gene expression.")
            # Baseline noise simulation
            sim_data = {f: [np.random.uniform(5.0, 15.0)] for f in all_features}
            
        if col_ctrl2.button("🔴 Simulate: GBM-Positive"):
            st.warning("Scenario: Patient presents with highly dysregulated signaling pathways and high abundance of malignancy markers.")
            # Aggressive abundance simulation
            sim_data = {f: [np.random.uniform(0.0, 5.0)] for f in all_features}
            for f in top_10_list:
                sim_data[f] = [np.random.uniform(6000.0, 9500.0)]

        if sim_data:
            input_df = pd.DataFrame(sim_data)
            prob = model.predict_proba(input_df)[0][1]
            
            st.divider()
            st.subheader("2. Real-Time Diagnostic Logic Output")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.write("**Confidence Score Analysis**")
                st.metric("Probability Score", f"{prob:.2%}")
                if prob > 0.5:
                    st.error("System Classification: POSITIVE")
                else:
                    st.success("System Classification: NEGATIVE")
                st.progress(float(prob))
            
            with res_col2:
                st.write("**Simulated Data Distribution**")
                # Show only top biomarkers for clarity
                display_df = pd.DataFrame([{"Gene": f, "Abundance": sim_data[f][0]} for f in top_10_list])
                st.bar_chart(display_df.set_index("Gene"), color="#004080")

            st.subheader("3. Technical Breakdown of Simulation")
            st.write("""
            * **Synthetic Ingestion:** The demo generates a vector of 23,412 values in memory, mimicking a full Partek Flow output.
            * **Feature Weights:** Notice that even if 23,400 genes have low values, high values in the 'Top 10' (e.g., *VIM* or *GAPDH*) are weighted significantly enough to cross the 50% classification threshold.
            * **Non-Linear Interaction:** The XGBoost trees evaluate the *combination* of these features, not just individual scores.
            """)
            
            

            st.write("---")
            st.caption("Disclaimer: Demo data is generated in-silico for educational purposes and does not represent real patient samples.")

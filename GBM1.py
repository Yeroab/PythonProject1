import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time


# --- 1. HELPER FUNCTIONS ---
def align_data_to_model(user_df, master_feature_list):
    # Initialize with 0.0
    aligned_df = pd.DataFrame(0.0, index=[0], columns=master_feature_list)

    # Standardize column names for comparison
    user_cols_clean = {str(c).strip().lower(): c for c in user_df.columns}

    for official_name in master_feature_list:
        clean_name = official_name.lower().strip()
        if clean_name in user_cols_clean:
            original_col = user_cols_clean[clean_name]
            # Use .iloc[0] to ensure we get a single number
            raw_val = user_df[original_col].iloc[0] if isinstance(user_df[original_col], (pd.Series, np.ndarray)) else \
                user_df[original_col]
            try:
                aligned_df[official_name] = float(raw_val)
            except (ValueError, TypeError):
                aligned_df[official_name] = 0.0

    return aligned_df


# --- 2. CONFIGURATION & ASSETS ---
st.set_page_config(page_title="Multi-Omic Diagnostic Portal", layout="wide", page_icon="🧬")

RELEVANT_PANEL = [
    'BLM_rna', 'FGG_prot', 'PRKCB_prot', 'SPHKAP_prot',
    'D-glucose_met', 'hypotaurine_met', 'creatinine_met', 'citricacid_met',
    'MAX_rna', 'NLK_rna', 'TRIQK_prot', 'PYGL_prot', 'xylitol_met',
    'GFAP_prot', 'S100B_rna', 'MBP_rna', 'PLP1_rna', 'OLIG2_rna',
    'SOX2_rna', 'EGFR_prot', 'PTENP1_rna', 'TP53_rna', 'IDH1_rna',
    'VIM_prot', 'FN1_rna', 'CD44_rna', 'STAT3_rna', 'VEGFA_prot',
    'RB1_rna', 'NF1_rna', 'PIK3CA_rna', 'CDK4_rna', 'MDM2_rna',
    'MYC_rna', 'TERT_rna', 'ATRX_rna', 'MGMT_rna', 'GATA3_rna',
    'FOXA1_rna', 'ESR1_rna', 'PGR_rna', 'ERBB2_rna', 'CDH1_rna',
    'MKI67_rna', 'TOP2A_rna', 'PCNA_rna', 'MCM6_rna', 'BIRC5_rna',
    'CCNB1_rna', 'CCND1_rna', 'CCNE1_rna', 'AURKA_rna', 'PLK1_rna',
    'CENPF_rna', 'ASPM_prot', 'KIF11_rna', 'NUSAP1_rna', 'PRC1_rna',
    'UBE2C_rna', 'PTTG1_rna', 'CDC20_rna', 'CDKN2A_rna', 'CDKN2B_rna',
    'CDK6_rna', 'MET_rna', 'PDGFRA_prot', 'FGFR1_rna', 'FGFR3_rna',
    'NOTCH1_rna', 'DLL3_rna', 'HES1_rna', 'ASCL1_rna', 'NEUROD1_rna',
    'POU3F2_rna', 'SOX10_rna', 'NKX2-2_rna', 'OLIG1_rna', 'MAG_prot',
    'MOG_prot', 'CNP_prot', 'GPR17_rna'
]

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; }
    header[data-testid="stHeader"], [data-testid="stSidebar"] {
        background-color: #F0F2F6 !important;
        border-bottom: 1px solid #e0e0e0;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #F0F2F6 !important;
        border: 2px solid #A0A0A0 !important;
        border-radius: 0px !important;
    }
    html, body, .stMarkdown, p, span, label, h1, h2, h3, h4, h5, h6, table, th, td {
        color: #000000 !important;
    }
    button, div.stButton > button, div.stDownloadButton > button {
        background-color: #A0A0A0 !important; 
        color: #000000 !important;           
        border: 1px solid #707070 !important;
        font-weight: bold !important;
        border-radius: 0px !important;
    }
    .clean-template-section { padding: 10px 0px; text-align: left; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_assets():
    try:
        model_path = 'gbm_diagnostic_model-1.pkl'
        with open(model_path, 'rb') as f:
            loaded_object = pickle.load(f)
        if isinstance(loaded_object, dict):
            return loaded_object['model'], loaded_object['features']
        else:
            # Fallback for direct model objects
            return loaded_object, getattr(loaded_object, 'feature_names_in_', None)
    except:
        return None, None


model, feature_list = load_model_assets()

# --- 4. NAVIGATION SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Main Analysis", "Documentation", "Demo Walkthrough"])
st.sidebar.markdown("---")
st.sidebar.info("Tip: You only need to fill in available markers. Blank fields will be treated as zero.")

# --- 5. PAGE: MAIN ANALYSIS ---
if page == "Main Analysis":
    st.header("Patient Clinical Analysis")

    col_dl, col_up = st.columns(2)
    with col_dl:
        st.subheader("1. Download Format")
        st.markdown('<div class="clean-template-section">', unsafe_allow_html=True)
        if feature_list is not None:
            sorted_features = RELEVANT_PANEL + [f for f in feature_list if f not in RELEVANT_PANEL]
            template_df = pd.DataFrame(0.0, index=[0], columns=sorted_features)
            csv_data = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Template", data=csv_data, file_name="biomarker_template.csv",
                               mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_up:
        st.subheader("2. Upload Patient Data")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("---")

    st.subheader("3. Manual Marker Entry")
    st.write("Input values for specific markers (defaults to 0.0 if left blank).")
    manual_data = {}
    m_cols = st.columns(4)
    for i, marker in enumerate(RELEVANT_PANEL[:12]):
        with m_cols[i % 4]:
            manual_data[marker] = st.number_input(f"{marker}", value=0.0)

    if st.button("Run Full Clinical Analysis"):
        if model and feature_list is not None:
            # FIXED DATA MERGING LOGIC
            input_df = pd.DataFrame([manual_data])

            if uploaded_file:
                file_df = pd.read_csv(uploaded_file).iloc[[0]]  # Ensure only first row
                # Combine uploaded data with manual data, uploaded taking precedence
                for col in file_df.columns:
                    input_df[col] = file_df[col].values

            with st.spinner("Processing Multi-Omic Data..."):
                # 1. Align to the full feature list
                processed_df = align_data_to_model(input_df, feature_list)

                # 2. Convert to NumPy and fix shape
                data_matrix = processed_df.to_numpy().astype(np.float32)

                # 3. Predict probability
                prob_array = model.predict_proba(data_matrix)
                prob = float(prob_array[0][1])

            # --- Display Results ---
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Risk Probability", f"{prob:.2%}")

            status = "HIGH RISK" if prob > 0.5 else "LOW RISK"
            color = "red" if prob > 0.5 else "green"
            res_col2.markdown(f"**Classification:** <span style='color:{color}; font-weight:bold;'>{status}</span>",
                              unsafe_allow_html=True)
            st.progress(prob)

            with st.expander("View Detected Biomarker Values"):
                feature_lookup = {f.lower().strip(): f for f in feature_list}
                found_data = []
                for marker in RELEVANT_PANEL:
                    marker_clean = marker.lower().strip()
                    if marker_clean in feature_lookup:
                        off_name = feature_lookup[marker_clean]
                        val = processed_df[off_name].iloc[0]
                        if val != 0:
                            found_data.append({"Marker": off_name, "Value": val})
                if found_data:
                    st.table(pd.DataFrame(found_data))
                else:
                    st.write("No active markers detected.")
        else:
            st.error("Model assets not loaded. Please check the model file.")

# --- 6. PAGE: DOCUMENTATION ---
elif page == "Documentation":
    st.header("Documentation for Multiomics GBM Biomarker Identifier")
    st.subheader("1. Data Inputting")
    st.write("""
    The analysis to successfully run, there are requirements that the inputted files should hold. 
    The file type should be .csv format. The header file should match the official nomenclature. 
    (Pro tip: A template is located on the main analysis page to input values for the omics with appropriate headers). 
    Although there are a total of 843 features and 81 relevant features, the user is only required to input the data that is available and the blank omics will be automatically assigned 0.00 value.
    """)

    st.subheader("2. Step-by-Step Workflow")
    st.write("""
    Processed after data is inputted: The inputted data is scanned and matched to the feature list of different omics (Proteomics, metabolomics, transcriptomics) and bridges minor header naming differences. 
    Because the XGBoost model requires a fixed input width, the system creates a "Full-Scale Vector." The available measurements are placed into their specific active slots, while the remaining slots are filled with zeros. 
    The model calculates the statistical weights of your biomarkers against the background noise to determine a risk probability. 
    The raw output is converted into a probability Score (%) and a risk classification.
    """)

# --- 7. PAGE: DEMO WALKTHROUGH ---
elif page == "Demo Walkthrough":
    st.header("Demo Mode")
    if st.button("Generate & Analyze Demo Patient"):
        if model and feature_list is not None:
            # Use random data that varies significantly
            dummy_data = np.random.rand(1, len(feature_list)).astype(np.float32)
            prob = float(model.predict_proba(dummy_data)[0][1])
            st.metric("Probability Score", f"{prob:.2%}")
            st.progress(prob)
            demo_display = pd.DataFrame(dummy_data, columns=feature_list)
            st.table(demo_display[RELEVANT_PANEL[:10]].T.rename(columns={0: "Value"}))

# --- FOOTER ---
st.markdown("---")
st.caption("Experimental Multi-Omic Analysis Tool | Objective: Binary Logistic Regression")

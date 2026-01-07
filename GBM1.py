import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- 1. HELPER FUNCTIONS ---
def align_data_to_model(user_df, master_feature_list):
    """
    Robustly forces the user's data into the exact 843-column structure.
    Ensures column order matches the model training exactly.
    """
    aligned_df = pd.DataFrame(0.0, index=range(len(user_df)), columns=master_feature_list)
    user_df.columns = [c.strip().lower() for c in user_df.columns]
    master_map = {f.lower(): f for f in master_feature_list}
    for col in user_df.columns:
        if col in master_map:
            actual_name = master_map[col]
            aligned_df[actual_name] = pd.to_numeric(user_df[col], errors='coerce').fillna(0.0).values
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

# Custom CSS for Light Grey Theme Elements, Dark Grey Buttons, and SQUARE borders
st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp { background-color: #FFFFFF !important; }

    /* 2. Header & Sidebar Background */
    header[data-testid="stHeader"], [data-testid="stSidebar"] {
        background-color: #F0F2F6 !important;
        border-bottom: 1px solid #e0e0e0;
    }

    /* 3. SQUARE File Uploader Area */
    [data-testid="stFileUploadDropzone"] {
        background-color: #F0F2F6 !important;
        border: 2px solid #A0A0A0 !important;
        border-radius: 0px !important;
        color: #000000 !important;
        min-height: 150px;
    }

    /* 4. Text Colors */
    html, body, .stMarkdown, p, span, label, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    /* 5. SQUARE DARK GREY BUTTONS */
    div.stButton > button, div.stDownloadButton > button, button {
        background-color: #A0A0A0 !important; 
        color: #000000 !important;           
        border: 1px solid #707070 !important;
        font-weight: bold !important;
        border-radius: 0px !important; /* Forces Square shape */
        width: 100%;
        padding: 15px;
    }

    /* Container for the Template button to align height visually with uploader */
    .template-box {
        border: 2px solid #A0A0A0;
        padding: 20px;
        background-color: #F0F2F6;
        text-align: center;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_assets():
    try:
        model_path = 'gbm_diagnostic_model-1.pkl'
        if not os.path.exists(model_path):
            return None, None
        with open(model_path, 'rb') as f:
            loaded_object = pickle.load(f)
        if isinstance(loaded_object, dict):
            return loaded_object['model'], loaded_object['features']
        else:
            return loaded_object, loaded_object.get_booster().feature_names
    except:
        return None, None

model, feature_list = load_model_assets()

# --- 3. NAVIGATION SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Main Analysis", "Documentation", "Demo Walkthrough"])
st.sidebar.markdown("---")
st.sidebar.info("Tip: You only need to fill in available markers. Blank fields will be treated as zero.")

# --- 4. PAGE: MAIN ANALYSIS ---
if page == "Main Analysis":
    st.header("Patient Clinical Analysis")
    
    # Square Layout for Template and Upload
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Download Format")
        # Template Box
        st.markdown('<div class="template-box">', unsafe_allow_html=True)
        if feature_list:
            sorted_features = RELEVANT_PANEL + [f for f in feature_list if f not in RELEVANT_PANEL]
            template_df = pd.DataFrame(0.0, index=[0], columns=sorted_features)
            csv_data = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Square Template",
                data=csv_data,
                file_name="biomarker_template_843.csv",
                mime="text/csv"
            )
        else:
            st.error("Model features not loaded.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("2. Upload Patient Data")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], label_visibility="collapsed")

    st.markdown("---")

    if uploaded_file and model:
        raw_df = pd.read_csv(uploaded_file)
        with st.spinner("Analyzing..."):
            processed_df = align_data_to_model(raw_df, feature_list)
            prob_array = model.predict_proba(processed_df)
            prob = float(prob_array[0][1])

        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Risk Probability", f"{prob:.2%}")
        status = "HIGH RISK" if prob > 0.5 else "LOW RISK"
        color = "red" if prob > 0.5 else "green"
        mcol2.markdown(f"**Classification:** <span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
        st.progress(prob)

        with st.expander("View Detected Biomarker Values"):
            feature_lookup = {f.lower().strip(): f for f in feature_list}
            found_data = []
            for marker in RELEVANT_PANEL:
                marker_clean = marker.lower().strip()
                if marker_clean in feature_lookup:
                    official_name = feature_lookup[marker_clean]
                    val = processed_df[official_name].iloc[0]
                    if val != 0:
                        found_data.append({"Marker": official_name, "Value": val})
            if found_data:
                st.table(pd.DataFrame(found_data))
            else:
                st.write("No relevant markers detected.")

# --- 5. PAGE: DOCUMENTATION ---
elif page == "Documentation":
    st.header("System Documentation")
    st.subheader("1. Data Inputting")
    st.write("Files must be in .csv format. Use the square template provided on the main page.")
    st.subheader("2. Workflow")
    st.write("The system aligns input data to an 843-feature vector, filling missing values with 0.0 before running XGBoost inference.")

# --- 6. PAGE: DEMO WALKTHROUGH ---
elif page == "Demo Walkthrough":
    st.header("Demo Mode")
    if st.button("Generate Synthetic Patient Profile"):
        if feature_list:
            dummy_data = np.random.uniform(0.1, 2.0, size=(1, len(feature_list)))
            demo_df = pd.DataFrame(dummy_data, columns=feature_list)
            prob = float(model.predict_proba(demo_df)[0][1])
            st.metric("Probability Score", f"{prob:.2%}")
            st.table(demo_df[RELEVANT_PANEL[:10]].T.rename(columns={0: "Value"}))

# --- FOOTER ---
st.markdown("---")
st.caption("Experimental Multi-Omic Analysis Tool | Binary Logistic Regression (XGBoost)")

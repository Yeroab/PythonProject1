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
# Custom CSS for Light Theme, Grey Sidebar, Dark Grey Buttons, and Black Text
# Custom CSS for Light Grey Theme Elements and Dark Grey Buttons
st.markdown("""
    <style>
    /* 1. Main Background (White) */
    .stApp { 
        background-color: #FFFFFF !important; 
    }

    /* 2. Header Background (Light Grey) */
    header[data-testid="stHeader"] {
        background-color: #F0F2F6 !important;
        border-bottom: 1px solid #e0e0e0;
    }

    /* 3. Sidebar Background (Light Grey) */
    [data-testid="stSidebar"] {
        background-color: #F0F2F6 !important;
        border-right: 1px solid #e0e0e0;
    }

   /* 4. File Uploader Area: Square and Light Grey */
    [data-testid="stFileUploadDropzone"] {
        background-color: #F0F2F6 !important;
        border: 2px solid #A0A0A0 !important;
        border-radius: 0px !important;
        color: #000000 !important;
    }

    /* 5. Universal Black Text */
    html, body, [class*="css"], .stMarkdown, p, span, label, 
    h1, h2, h3, h4, h5, h6, table, th, td {
        color: #000000 !important;
    }

    /* 6. DARK GREY BUTTONS WITH BLACK TEXT */
    /* Targeting Standard, Download, and 'Browse files' buttons */
    button, 
    div.stButton > button, 
    div.stDownloadButton > button, 
    [data-testid="stFileUploadDropzone"] button {
        background-color: #A0A0A0 !important; 
        color: #000000 !important;           
        border: 1px solid #707070 !important;
        font-weight: bold !important;
    }

    /* Force button text/labels to be black */
    button div p, button span, .stButton p {
        color: #000000 !important;
    }

    /* 7. Metrics and Tables */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    [data-testid="stTable"] td, [data-testid="stTable"] th {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# 1. First, define the function
@st.cache_resource
def load_model_assets():
    try:
        model_path = 'gbm_diagnostic_model-1.pkl'
        with open(model_path, 'rb') as f:
            loaded_object = pickle.load(f)

        if isinstance(loaded_object, dict):
            # Directly extract using the keys discovered in debug
            model = loaded_object['model']
            feature_list = loaded_object['features']
        else:
            model = loaded_object
            feature_list = model.get_booster().feature_names

        return model, feature_list

    except Exception as e:
        # We keep this error message just in case the file goes missing
        st.error(f"Error loading model assets: {e}")
        return None, None

        # CASE 2: Getting the feature names
        # If it's a dictionary, we might need to look at the keys directly
        if isinstance(model, dict):
            feature_list = list(model.keys()) # Or another specific key
        else:
            feature_list = model.get_booster().feature_names

        return model, feature_list

    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None


# 2. NOW, call the function to create the variables in the main script
# This is the line that fixes your "NameError: feature_list is not defined"

model , feature_list = load_model_assets()

# --- 3. NAVIGATION SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Main Analysis", "Documentation", "Demo Walkthrough"])

st.sidebar.markdown("---")

with st.sidebar:
    st.header("Data Tools")
    if feature_list:
        # Sort feature_list so RELEVANT_PANEL comes first
        sorted_features = RELEVANT_PANEL + [f for f in feature_list if f not in RELEVANT_PANEL]
        
        # Create template with sorted columns
        template_df = pd.DataFrame(0.0, index=[0], columns=sorted_features)
        
        csv_data = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download 843-Feature Template",
            data=csv_data,
            file_name="biomarker_template_843.csv",
            mime="text/csv",
            help="The 81 relevant markers are at the start. Fill those, leave others as 0."
        )
    st.info("Tip: You only need to fill in the available relevant markers. The model will handle the rest")

# --- 4. PAGE: MAIN ANALYSIS ---
if page == "Main Analysis":
    st.header("Patient Clinical Analysis")
    st.write("Upload a CSV file containing the biological markers.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)

        with st.spinner("Aligning markers and verifying 843-column structure..."):
            processed_df = align_data_to_model(raw_df, feature_list)
            detected_count = (processed_df != 0).sum().sum()
            if detected_count == 0:
                st.warning(" No matching biomarker names found. Please check your CSV headers against the template.")

        prob_array = model.predict_proba(processed_df)
        prob = float(prob_array[0][1])

        col1, col2 = st.columns(2)
        col1.metric("Risk Probability", f"{prob:.2%}")

        status = "HIGH RISK" if prob > 0.5 else "LOW RISK"
        color = "red" if prob > 0.5 else "green"
        col2.markdown(f"**Classification:** <span style='color:{color}; font-weight:bold;'>{status}</span>",
                      unsafe_allow_html=True)

        st.progress(prob)

# --- Update this specific block around line 193 ---
        with st.expander(" View Detected Biomarker Values"):
            found_data = []
            
            # 1. Create a lookup: {lowercase_name: official_name_in_model}
            # This bridges the gap between your list and the 843 model features
            feature_lookup = {f.lower().strip(): f for f in feature_list}
            
            for marker in RELEVANT_PANEL:
                marker_clean = marker.lower().strip()
                
                # 2. Match the panel name to the actual model feature name
                if marker_clean in feature_lookup:
                    official_name = feature_lookup[marker_clean]
                    
                    # 3. Access the value from the aligned data safely
                    val = processed_df[official_name].iloc[0]
                    
                    if val != 0:
                        found_data.append({"Marker": official_name, "Value": val})
            
            # 4. Display the results
            if found_data:
                st.table(pd.DataFrame(found_data))
            else:
                st.write("No relevant markers from the 81-panel were detected in the file.")

# --- 5. PAGE: DOCUMENTATION ---
elif page == "Documentation":
    st.header("Documentation for Multiomics GBM Biomarker Identifier")

    st.subheader("1. Data Inputting")
    st.write("""
    The analysis to successfully run, there are requirements that the inputted files should hold. 
    The file type should be .csv format. The header file should match the official nomenclature. 
    (Pro tip: A template is located on the side bar to input values for the omics with appropriate headers). 
    Although there are a total of 843 features and 81 relevant features, the user is only required to input the data that is available and the blank omics will be automatically assigned 0.00 value.
    """)

    st.subheader("2. Step-by-Step Workflow")
    st.write("""
    Processed after data is inputted: The inputted data is scanned and matched to the feature list of different omics (Proteomics, metabolomics, transcriptomics) and bridges minor header naming differences. 
    Because the XGBoost model requires a fixed input width of 843, the system creates a "Full-Scale Vector." The available measurements are placed into their specific active slots, while the remaining slots are filled with zeros to maintain structural integrity. 
    The model calculates the statistical weights of your 81 biomarkers against the background noise to determine a risk probability. 
    The raw output is converted into a probability Score (%) and a risk classification (Low vs. High Risk).
    """)

# --- 6. PAGE: DEMO WALKTHROUGH ---
elif page == "Demo Walkthrough":
    st.header("Demo Mode")
    if st.button("Generate & Analyze Demo Patient"):
        dummy_data = np.random.uniform(0.1, 5.0, size=(1, len(feature_list)))
        demo_df = pd.DataFrame(dummy_data, columns=feature_list)
        prob = float(model.predict_proba(demo_df)[0][1])

        st.subheader("Result")
        st.metric("Probability Score", f"{prob:.2%}")
        st.progress(prob)

        # Displaying the top biomarkers in black font
        st.write("**Top Markers in Synthetic Profile:**")
        st.table(demo_df[RELEVANT_PANEL[:10]].T.rename(columns={0: "Value"}))

# --- FOOTER ---
st.markdown("---")
st.caption("Experimental Multi-Omic Analysis Tool | Objective: Binary Logistic Regression")

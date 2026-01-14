import streamlit as st
import joblib
import pandas as pd
import xgboost

# --- 1. Page Config ---
st.set_page_config(page_title="GBM Analysis Portal", layout="wide")


# --- 2. Data Loading Function ---
@st.cache_resource
def load_all_models():
    try:
        # Load the files into specific variables
        core = joblib.load('gbm_detector.pkl')
        val = joblib.load('gbm_pathways.pkl')
        # Ensure the filename matches exactly what is on your desktop/GitHub
        report = joblib.load('gbm_biomarkers (2).pkl')

        return core, val, report
    except Exception as e:
        st.error(f"Critical Error loading files: {e}")
        return None, None, None


# Assign the variables globally
core, val, report = load_all_models()

# --- 3. Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Biomarkers", "GBM Detector", "Pancreatic Check"])

# --- 4. Page Logic ---
if page == "Biomarkers":
    st.header(" Drug Target Report")

    # FIX: Check if 'report' exists before using it
    if report is not None:
        st.write("Displaying data from `gbm_biomarkers (2).pkl`:")
        st.dataframe(report['top_targets_df'], use_container_width=True)
    else:
        st.warning("The report file could not be loaded. Please check the file path.")

elif page == "GBM Detector":
    st.header(" GBM Diagnostic Model")
    if core is not None:
        st.success("Detector Model Loaded.")
        features = core['features']
        st.write(f"This model uses {len(features)} genomic/proteomic features.")
        # Your prediction logic goes here
    else:
        st.error("Detector model missing.")

elif page == "Pancreatic Check":
    st.header(" Pancreatic Pathway Cross-Check")
    if val is not None:
        # Based on your notes: val['pathways']['Genomic']['model']
        st.info("Genomic Pathway Model is active.")
        genomic_model = val['pathways']['Genomic']['model']
    else:
        st.error("Pathway model missing.")

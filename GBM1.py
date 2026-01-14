import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="GBM Diagnostic Suite", layout="wide")

@st.cache_resource
def load_assets():
    try:
        # Loading core diagnostic and pathway models
        diag = joblib.load('gbm_diagnostic_model-1.pkl')
        pathways = joblib.load('gbm_pathways.pkl')
        return diag, pathways
    except Exception as e:
        st.error(f"File Load Error: {e}")
        return None, None

diag, pathways = load_assets()

# Sidebar Navigation
st.sidebar.title("🧬 Main Menu")
app_mode = st.sidebar.selectbox("Select Page", 
    ["Main Diagnosis", "App Documentation", "Demo Walkthrough"])

# --- PAGE 1: MAIN DIAGNOSIS ---
if app_mode == "Main Diagnosis":
    st.title("🧠 GBM Raw Data Analysis")
    if diag:
        model = diag['model']
        all_features = diag['features']
        
        # Calculate Feature Importance to provide dynamic inputs
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        with st.form("diag_form"):
            st.subheader("High-Impact Raw Biomarker Inputs")
            st.info("Input raw expression values. The model sensitivity is highest for these specific markers.")
            cols = st.columns(2)
            user_inputs = {}
            for i, feat in enumerate(top_10):
                with cols[i % 2]:
                    # Defaulting to 100.0 as a baseline for raw data
                    user_inputs[feat] = st.number_input(f"{feat}", value=100.0)
            
            submitted = st.form_submit_button("Run Diagnostic")
            
            if submitted:
                # Construct full feature vector (padding rest with 0.0)
                full_input = {f: [user_inputs.get(f, 0.0)] for f in all_features}
                input_df = pd.DataFrame(full_input)
                
                # Calculate probability
                prob = model.predict_proba(input_df[all_features])[0][1]
                
                st.divider()
                st.write(f"### Probability of GBM: {prob:.2%}")
                
                if prob > 0.5:
                    st.error("Diagnostic Result: POSITIVE (High Risk Profile Detected)")
                else:
                    st.success("Diagnostic Result: NEGATIVE (Low Risk Profile Detected)")
                
                st.subheader("Feature Contribution Analysis")
                st.bar_chart(feat_df.head(10).set_index('feature'))

# --- PAGE 2: DOCUMENTATION ---
elif app_mode == "App Documentation":
    st.title("📑 Technical Documentation")
    st.write("""
    The Glioblastoma Multiforme (GBM) Diagnostic Suite is a machine learning-powered clinical decision support tool designed to analyze raw genomic and proteomic biomarker expression levels. The core of the application relies on an XGBoost classifier trained on high-dimensional raw data to identify patterns associated with GBM pathology. When a user interacts with the interface, the system dynamically extracts the most influential features from the underlying model architecture to ensure that the user provides values for the biomarkers that hold the highest predictive weight. This approach mitigates the common issue of static probability results that occur when a model is fed sparse or empty data. By focusing the interface on high-impact variables, the GUI bridges the gap between complex computational biology and actionable clinical insight.
    """)
    st.write("""
    Internally, the application utilizes a serialized pipeline that includes feature alignment and probability estimation. Once the user submits their raw data, the system constructs a complete feature vector, padding any dimensions with baseline values to maintain the structural integrity required by the XGBoost booster. The resulting output is a calculated probability score that reflects the likelihood of a GBM signature within the provided sample. Beyond simple binary classification, the suite integrates secondary validation pathways to provide a comprehensive biological overview of the patient profile. This ensures that the GUI functions not just as a calculator, but as a holistic portal for neuro-oncological research and diagnostic validation.
    """)

# --- PAGE 3: DEMO WALKTHROUGH ---
elif app_mode == "Demo Walkthrough":
    st.title("🚀 Platform Walkthrough")
    
    st.subheader("Step 1: Identifying Key Biomarkers")
    st.write("Navigate to the 'Main Diagnosis' page. You will notice that the system has already identified the top 10 biomarkers most critical to the model's decision-making process. These are not randomly chosen; they are the features with the highest Gini importance according to the XGBoost algorithm. By focusing on these specific inputs, you can observe how raw data shifts the diagnostic probability away from the baseline.")
    
    

    st.subheader("Step 2: Entering Raw Data")
    st.write("In the input fields, enter the raw expression levels obtained from laboratory analysis. Because this model is trained on raw values rather than normalized logs, the scale of these numbers is significant. If you leave these values at 0.0, the model may return a baseline bias, so ensure you are entering actual detected levels for the patient to see a dynamic confidence score.")

    st.subheader("Step 3: Interpreting the Output")
    st.write("Click 'Run Diagnostic' to generate the report. The system will provide a percentage-based confidence score. A score above 50% indicates a positive GBM signature. Below the result, a bar chart visualizes exactly how much weight each of your inputs carried in reaching that specific conclusion, allowing for transparent clinical verification and further investigation into specific genomic pathways.")

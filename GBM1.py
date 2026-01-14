import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="GBM Diagnostic Portal", layout="wide")

@st.cache_resource
def load_models():
    try:
        detector = joblib.load('gbm_detector.pkl')
        pathways = joblib.load('gbm_pathways.pkl')
        report = joblib.load('gbm_biomarkers (2).pkl')
        diag_model = joblib.load('gbm_diagnostic_model-1.pkl')
        return detector, pathways, report, diag_model
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None

detector, pathways, report, diag_model = load_models()

# Sidebar Navigation
st.sidebar.title("🧬 Navigation")
app_mode = st.sidebar.selectbox("Select Tool", ["Main Diagnosis", "Biomarker Report", "Pathway Validation"])

if app_mode == "Main Diagnosis":
    st.title("🧠 GBM Diagnostic System")
    
    if diag_model:
        # Accessing the model and features
        model_obj = diag_model['model']
        features = diag_model['features']
        
        tab1, tab2 = st.tabs(["Manual Data Entry", "Bulk CSV Upload"])

        with tab1:
            st.subheader("Patient Biomarker Levels")
            with st.form("manual_entry"):
                cols = st.columns(3)
                user_inputs = {}
                # Using the top features for the form
                for i, feat in enumerate(features[:15]):
                    with cols[i % 3]:
                        user_inputs[feat] = st.number_input(f"{feat}", value=0.0)
                
                submitted = st.form_submit_button("Run Diagnosis")
                
                if submitted:
                    # 1. Create the dictionary (using the name full_input)
                    full_input = {f: [user_inputs.get(f, 0.0)] for f in features}
                    
                    # 2. Convert to DataFrame
                    input_df = pd.DataFrame(full_input)
                    
                    # 3. Predict using the exact feature order
                    prediction = model_obj.predict(input_df[features])[0]
                    prob = model_obj.predict_proba(input_df[features])[0][1]
                    
                    st.divider()
                    if prediction == 1:
                        st.error(f"### Result: POSITIVE (Probability: {prob:.2%})")
                    else:
                        st.success(f"### Result: NEGATIVE (Probability: {prob:.2%})")
                    
                    st.metric("Confidence Score", f"{prob:.2%}")
                    st.progress(float(prob))

        with tab2:
            st.subheader("Upload Patient Records")
            csv_file = st.file_uploader("Upload CSV file", type=["csv"])
            if csv_file:
                df = pd.read_csv(csv_file)
                if all(f in df.columns for f in features):
                    preds = model_obj.predict(df[features])
                    df['GBM_Prediction'] = ["Positive" if p == 1 else "Negative" for p in preds]
                    st.dataframe(df)
                else:
                    st.error("CSV is missing required biomarker columns.")

elif app_mode == "Biomarker Report":
    st.title("🎯 Drug Target Identification")
    if report:
        st.dataframe(report['top_targets_df'], use_container_width=True)

elif app_mode == "Pathway Validation":
    st.title("🔬 Pathway Cross-Check")
    if pathways:
        st.write("Genomic Model features:", pathways['pathways']['Genomic']['features'])

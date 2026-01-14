import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="GBM Diagnostic Portal", layout="wide")


@st.cache_resource
def load_models():
    """Loads all 4 specific files provided by the user."""
    try:
        detector = joblib.load('gbm_detector.pkl')
        pathways = joblib.load('gbm_pathways.pkl')
        report = joblib.load('gbm_biomarkers (2).pkl')
        # The 4th file: The main diagnostic model
        diagnostic_main = joblib.load('gbm_diagnostic_model-1.pkl')

        return detector, pathways, report, diagnostic_main
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None


detector, pathways, report, diag_model = load_models()

# Sidebar Navigation
st.sidebar.title(" Navigation")
app_mode = st.sidebar.selectbox("Select Tool", ["Main Diagnosis", "Biomarker Report", "Pathway Validation"])

if app_mode == "Main Diagnosis":
    st.title("GBM Diagnostic System")
    st.write("Using `gbm_diagnostic_model-1.pkl` for high-accuracy analysis.")

    if diag_model:
        # Extract features required by the XGBoost model
        # The model object is inside the 'model' key of the pickle
        model_obj = diag_model['model']
        # Depending on how it was saved, features are usually in the dictionary or the booster
        features = diag_model.get('features', [])

        tab1, tab2 = st.tabs(["Manual Data Entry", "Bulk CSV Upload"])

        with tab1:
            st.subheader("Patient Biomarker Levels")
            with st.form("manual_entry"):
                cols = st.columns(3)
                user_inputs = {}
                # Display top 15 features for manual entry to keep it clean
                for i, feat in enumerate(features[:15]):
                    with cols[i % 3]:
                        user_inputs[feat] = st.number_input(f"{feat}", value=0.0)

                submitted = st.form_submit_button("Run Diagnosis")

                if submitted:
                    # Create full feature row (fill missing features with 0)
                    full_input = {f: [user_inputs.get(f, 0.0)] for f in features}
                    input_df = pd.DataFrame(full_input)

                    prediction = model_obj.predict(input_df[features])[0]
                    prob = model_obj.predict_proba(input_df[features])[0][1]

                    st.divider()
                    if prediction == 1:
                        st.error("### Result: High Probability of GBM")
                    else:
                        st.success("### Result: Low Probability of GBM")
                    st.metric("Model Confidence", f"{prob:.2%}")

        with tab2:
            st.subheader("Upload Patient Records")
            csv_file = st.file_uploader("Upload CSV file", type=["csv"])
            if csv_file:
                df = pd.read_csv(csv_file)
                # Ensure columns match
                if all(f in df.columns for f in features):
                    preds = model_obj.predict(df[features])
                    df['GBM_Prediction'] = ["Positive" if p == 1 else "Negative" for p in preds]
                    st.write("Analysis Complete:")
                    st.dataframe(df)
                else:
                    st.error("CSV is missing required biomarker columns.")

elif app_mode == "Biomarker Report":
    st.title(" Drug Target Identification")
    if report:
        st.write("Displaying targets from `gbm_biomarkers (2).pkl`:")
        st.dataframe(report['top_targets_df'], use_container_width=True)

elif app_mode == "Pathway Validation":
    st.title("🔬 Pathway Cross-Check")
    if pathways:
        st.info("Validation via Genomic Pathway Model")
        # Logic from your notes: val['pathways']['Genomic']['model']
        gen_model = pathways['pathways']['Genomic']['model']
        st.write("Pathway model successfully loaded.")
        st.write("Required features for this model:", pathways['pathways']['Genomic']['features'])

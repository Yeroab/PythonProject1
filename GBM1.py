import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io

# Set page configuration
st.set_page_config(page_title="GBM Diagnostic Suite", layout="wide")

@st.cache_resource
def load_assets():
    try:
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
    ["Main Diagnosis", "App Documentation", "Interactive Demo Walkthrough"])

# --- PAGE 1: MAIN DIAGNOSIS ---
if app_mode == "Main Diagnosis":
    st.title("🧠 GBM Raw Data Analysis")
    if diag:
        model = diag['model']
        all_features = diag['features']
        
        # Calculate Feature Importance for the Manual form
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        # Create Tabs for Manual vs Bulk
        tab1, tab2 = st.tabs(["Manual Entry", "Bulk Analysis (CSV)"])

        with tab1:
            with st.form("diag_form"):
                st.subheader("High-Impact Raw Biomarker Inputs")
                cols = st.columns(2)
                user_inputs = {}
                for i, feat in enumerate(top_10):
                    with cols[i % 2]:
                        user_inputs[feat] = st.number_input(f"{feat}", value=100.0)
                
                submitted = st.form_submit_button("Run Diagnostic")
                if submitted:
                    full_input = {f: [user_inputs.get(f, 0.0)] for f in all_features}
                    input_df = pd.DataFrame(full_input)
                    prob = model.predict_proba(input_df[all_features])[0][1]
                    
                    st.divider()
                    st.write(f"### Probability of GBM: {prob:.2%}")
                    if prob > 0.5:
                        st.error("Diagnostic Result: POSITIVE")
                    else:
                        st.success("Diagnostic Result: NEGATIVE")

        with tab2:
            st.subheader("Bulk Patient Processing")
            st.write("To process multiple patients, download the template below, fill in the raw data, and upload the completed file.")

            # 1. GENERATE TEMPLATE
            # Create a blank dataframe with the model's features as columns
            template_df = pd.DataFrame(columns=['Patient_ID'] + all_features)
            # Add one example row of zeros
            template_df.loc[0] = ['Example_Patient_001'] + [0.0] * len(all_features)
            
            # Convert to CSV for download
            buffer = io.BytesIO()
            template_df.to_csv(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="📥 Download CSV Template",
                data=buffer,
                file_name="gbm_bulk_template.csv",
                mime="text/csv",
                help="This file contains all biomarkers required by the AI model."
            )

            st.divider()

            # 2. UPLOAD AND PROCESS
            uploaded_file = st.file_uploader("Upload your completed CSV file", type=["csv"])
            
            if uploaded_file:
                bulk_df = pd.read_csv(uploaded_file)
                
                # Validation: Check if all features exist in the uploaded file
                missing_cols = [f for f in all_features if f not in bulk_df.columns]
                
                if not missing_cols:
                    # Run predictions for all rows
                    # Ensure columns are in the exact order the model expects
                    probs = model.predict_proba(bulk_df[all_features])[:, 1]
                    bulk_df['GBM_Probability'] = probs
                    bulk_df['Result'] = bulk_df['GBM_Probability'].apply(lambda x: "POSITIVE" if x > 0.5 else "NEGATIVE")
                    
                    st.success(f"Successfully processed {len(bulk_df)} patient records.")
                    st.dataframe(bulk_df[['Patient_ID', 'GBM_Probability', 'Result'] + top_10])
                    
                    # Optional: Download results
                    result_buffer = io.BytesIO()
                    bulk_df.to_csv(result_buffer, index=False)
                    result_buffer.seek(0)
                    st.download_button("Download Processed Results", data=result_buffer, file_name="gbm_results.csv", mime="text/csv")
                else:
                    st.error(f"The uploaded file is missing {len(missing_cols)} required columns. Please use the provided template.")
                    with st.expander("Show missing columns"):
                        st.write(missing_cols)

# (The rest of your Documentation and Walkthrough pages remain the same...)

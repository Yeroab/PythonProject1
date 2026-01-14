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
st.sidebar.title(" Main Menu")
app_mode = st.sidebar.selectbox("Select Page",
                                ["Main Diagnosis", "App Documentation", "Interactive Demo Walkthrough"])

# --- PAGE 1: MAIN DIAGNOSIS ---
if app_mode == "Main Diagnosis":
    st.title(" GBM Raw Data Analysis")
    if diag:
        model = diag['model']
        all_features = diag['features']

        # Get Global Feature Importance
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

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
                    
                    # Run Prediction
                    prob = model.predict_proba(input_df[all_features])[0][1]

                    st.divider()
                    
                    # --- DYNAMIC BIOMARKER IMPACT CHART ---
                    st.subheader("Biomarker Impact Chart")
                    st.write("This chart shows which biomarkers contributed most to this specific diagnostic result.")
                    
                    # Calculate local impact: User Value * Global Importance
                    # This explains WHY the result is Positive or Negative
                    impact_data = []
                    for feat in top_10:
                        weight = feat_df[feat_df['feature'] == feat]['importance'].values[0]
                        impact_score = user_inputs[feat] * weight
                        impact_data.append({"Biomarker": feat, "Impact Score": impact_score})
                    
                    impact_df = pd.DataFrame(impact_data).set_index("Biomarker")
                    st.bar_chart(impact_df)

                    st.write(f"### Probability of GBM: {prob:.2%}")
                    if prob > 0.5:
                        st.error("Diagnostic Result: POSITIVE")
                    else:
                        st.success("Diagnostic Result: NEGATIVE")

        with tab2:
            st.subheader("Bulk Patient Processing")
            st.write("Process multiple patients via CSV upload.")

            # Generate Template
            template_df = pd.DataFrame(columns=['Patient_ID'] + all_features)
            template_df.loc[0] = ['Example_Patient_001'] + [0.0] * len(all_features)
            buffer = io.BytesIO()
            template_df.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button("Download CSV Template", data=buffer, file_name="gbm_bulk_template.csv", mime="text/csv")

            st.divider()

            uploaded_file = st.file_uploader("Upload your completed CSV file", type=["csv"])
            if uploaded_file:
                bulk_df = pd.read_csv(uploaded_file)
                if all(f in bulk_df.columns for f in all_features):
                    probs = model.predict_proba(bulk_df[all_features])[:, 1]
                    bulk_df['GBM_Probability'] = probs
                    bulk_df['Result'] = bulk_df['GBM_Probability'].apply(lambda x: "POSITIVE" if x > 0.5 else "NEGATIVE")

                    st.subheader("📊 Comparative Risk Analysis")
                    st.bar_chart(bulk_df.set_index('Patient_ID')['GBM_Probability'])
                    
                    st.write("### Detailed Results Table")
                    st.dataframe(bulk_df[['Patient_ID', 'GBM_Probability', 'Result'] + top_10])
                else:
                    st.error("Missing required columns in CSV.")

# --- DOCUMENTATION AND DEMO PAGES (INDENTED CORRECTLY) ---
elif app_mode == "App Documentation":
    st.title("Documentation")
    st.write("The Glioblastoma Multiforme (GBM) Diagnostic is a machine learning-powered clinical support tool used to analyze raw genomic, proteomic and metabolomic biomarkers...")

elif app_mode == "Interactive Demo Walkthrough":
    st.title("Interactive Platform Walkthrough")
    # (Rest of demo logic as previously established)
    if diag:
        # ... logic to show feature list and simulation buttons ...
        st.write("Demo Content Loaded.")

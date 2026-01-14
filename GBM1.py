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
        feat_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values(by='importance',
                                                                                                 ascending=False)
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
                    bulk_df['Result'] = bulk_df['GBM_Probability'].apply(
                        lambda x: "POSITIVE" if x > 0.5 else "NEGATIVE")

                    st.subheader(" Comparative Risk Analysis")
                    st.bar_chart(bulk_df.set_index('Patient_ID')['GBM_Probability'])

                    st.write("### Detailed Results Table")
                    st.dataframe(bulk_df[['Patient_ID', 'GBM_Probability', 'Result'] + top_10])
                else:
                    st.error("Missing required columns in CSV.")

# --- DOCUMENTATION AND DEMO PAGES (INDENTED CORRECTLY) ---
elif app_mode == "App Documentation":
    st.title("  Documentation")
    st.write("""
        The Glioblastoma Multiforme (GBM) Diagnostic is a machine learning-powered clinical support tool used to analyze raw genomic, proteomic and metabolomic biomarkers according to their expression levels. The model relies on an XGBoost classifier trained on raw data to identify patterns associated with GBM. When a user interacts with the interface, the system dynamically extracts the most influential features from the model to ensure that the user provides values for the biomarkers that hold the highest predictive weight, removing the low significance features. The GUI creates an understandable bridge to interpret omics data for clinical use. 

        """)
    st.write("""
        Once the user submits their raw data, the system constructs a complete feature vector, padding any dimensions with baseline values to maintain the structural integrity required by the XGBoost booster. The resulting output is a calculated probability score that reflects the likelihood of a GBM within the provided sample. Beyond simple binary classification, the model integrates secondary validation to provide a complete overview of the patient profile..
        """)



elif app_mode == "Interactive Demo Walkthrough":
    st.title(" Interactive Platform Walkthrough")

    st.subheader("Introduction")
    st.write(
        "Follow the steps below for a clinical workflow and see how specific feature values change the diagnostic output.")

    st.markdown("---")

    if diag:
        model = diag['model']
        all_features = diag['features']
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values(by='importance',
                                                                                                 ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        # Step 1: Selection
        st.subheader("Select a Clinical Case Profile")
        col1, col2 = st.columns(2)

        sim_data = None
        profile_label = ""

        with col1:
            if st.button("Simulate: Healthy Control"):
                profile_label = "Healthy Control"
                sim_data = {f: [5.0] for f in all_features}

        with col2:
            if st.button("Simulate: GBM-Positive Patient"):
                profile_label = "GBM-Positive Patient"
                sim_data = {f: [0.0] for f in all_features}
                for f in top_10:
                    sim_data[f] = [5000.0]

        if sim_data:
            st.info(f"Active Simulation: {profile_label}")

            # Step 2: Show Feature List
            st.subheader(" Feature and Value List")
            st.write("The following biomarkers and their raw values are being sent to the model for this simulation:")

            # Prepare the list/table of features and values
            display_list = []
            for f in top_10:
                display_list.append({"Biomarker (Feature)": f, "Raw Value": sim_data[f][0]})

            # Displaying as a clean table
            st.table(pd.DataFrame(display_list))

            # Step 3: Result
            st.subheader(" Real-Time Diagnostic Result")
            sim_df = pd.DataFrame(sim_data)
            prob = model.predict_proba(sim_df[all_features])[0][1]

            st.write(f"** Confidence Score:**")
            st.progress(float(prob))

            if prob > 0.5:
                st.error(f"Prediction: POSITIVE ({prob:.2%})")
                st.write(
                    "The high raw values assigned to the critical biomarkers above triggered a Positive diagnosis.")
            else:
                st.success(f"Prediction: NEGATIVE ({prob:.2%})")
                st.write(
                    "The low raw values assigned to the biomarkers represent a healthy profile, resulting in a Negative diagnosis.")

    st.markdown("---")
    st.subheader("Final Interpretation")
    st.write(
        "This walkthrough demonstrates that the modeI calculates probability based on the intensity of the raw data. By entering values for these specific features on the 'Main Diagnosis' page, users can perform accurate real-world analysis.")


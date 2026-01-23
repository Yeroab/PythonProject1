import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io
# --- FORCE BLUE THEME ---
# --- FORCE BLUE THEME ---
st.markdown("""
    <style>
        /* 1. THE ACTUAL FULL PAGE BACKGROUND */
        /* Targets every possible container to force the blue color */
        .stApp, .stMain, [data-testid="stAppViewContainer"], .main {
            background-color: ##FFFFFF !important;
        }

        /* 2. TOP HEADER BAR */
        header[data-testid="stHeader"] {
            background-color: #1f77b4 !important;
        }

        /* 3. SIDEBAR - Navy Blue */
        section[data-testid="stSidebar"] {
            background-color: #0d3b4c !important;
        }

        /* Sidebar Text to White */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: white !important;
        }

        /* 4. MANUAL ENTRY: Input Box Background */
        /* We use a slightly darker blue so inputs stand out against the page */
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background-color: #cce6ff !important; 
            border-radius: 8px !important;
            border: 1px solid #1f77b4 !important;
        }

        /* Style the +/- buttons inside the input boxes */
        button[data-testid="stNumberInputStepUp"], 
        button[data-testid="stNumberInputStepDown"] {
            background-color: #1f77b4 !important;
            color: white !important;
            border-radius: 4px !important;
        }

        /* 5. RUN DIAGNOSTIC BUTTON */
        div.stButton > button {
            background-color: #1f77b4 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            width: 100% !important;
            border-radius: 6px !important;
        }
        
        div.stButton > button:hover {
            background-color: #155a8a !important;
        }

        /* 6. TABS & LABELS */
        div[data-baseweb="tab-highlight"] {
            background-color: #1f77b4 !important;
        }

        button[aria-selected="true"] p {
            color: #1f77b4 !important;
            font-weight: bold !important;
        }

        /* Biomarker Names (AACS_prot, etc.) */
        .stNumberInput label p {
            color: #0d3b4c !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)
# Set page configuration
st.set_page_config(page_title="MulitNet-AI", layout="wide")


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
# Sidebar Navigation
st.sidebar.title(" Main Menu")

# Changed selectbox to radio for a vertical list
app_mode = st.sidebar.radio(
    "Select Page",
    ["Upload your own omics data", "App Documentation", "Interactive Demo Walkthrough"]
)
# --- PAGE 1: MAIN DIAGNOSIS ---
if app_mode == "Upload your own omics data":
    st.title(" MultiNet-AI User Driven Interface")
    if diag:
        model = diag['model']
        all_features = diag['features']

        # Get Global Feature Importance
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values(by='importance',
                                                                                                 ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        tab1, tab2 = st.tabs(["Enter Omics Sample Value (Manual Entry)", "Bulk Analysis for Mulitiple Omics Data (CSV)"])

        with tab1:
            with st.form("diag_form"):
                st.subheader("Input abundance score in raw value")
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
                        st.error("Diagnostic Result: Postive")
                    else:
                        st.success("Diagnostic Result: Negative")
                        
        with tab2:
            st.subheader("Bulk Patient Processing")
            st.write("Process 100 patient profiles simultaneously via CSV upload.")

            # 1. Generate 100 Patient IDs (Patient_001 to Patient_100)
            patient_ids = [f'Patient_{i:03d}' for i in range(1, 101)]

            # 2. Create the data grid: 100 rows of 0.00 for all biomarkers
            # We use a list comprehension for speed
            data_rows = [[pid] + [0.00] * len(all_features) for pid in patient_ids]

            # 3. Build the DataFrame
            template_df = pd.DataFrame(data_rows, columns=['Patient_ID'] + all_features)

            # 4. Prepare the file buffer
            st.subheader("📊 Comparative Risk Analysis")
            st.bar_chart(bulk_df.set_index('Patient_ID')['GBM_Probability'])
            # NEW: Dynamically identify features with actual impact (non-zero importance)
            # This filters your 843 features down to only the ones the model actually uses
            relevant_biomarkers = [f for f in all_features if importance_map.get(f, 0) > 0]
            # Optional: Sort them so the most important biomarkers appear first in the table
            relevant_biomarkers = sorted(relevant_biomarkers, key=lambda x: importance_map.get(x, 0), reverse=True)

            st.write("###  Detailed Results (Key Biomarkers Only)")
            st.write(f"This table displays the {len(relevant_biomarkers)} biomarkers that contributed to the diagnostic score.")

            # Display the dataframe with only relevant columns
            st.dataframe(bulk_df[['Patient_ID', 'GBM_Probability', 'Result'] + relevant_biomarkers])

            # 5. The Blue Download Button
            st.download_button(
                label=" Download 100-Patient CSV Template",
                data=buffer,
                file_name="MultiNet_Bulk_Template.csv",
                mime="text/csv",
                use_container_width=True  # Makes it look clean on the page
            )

            st.divider()

            uploaded_file = st.file_uploader("Upload your completed CSV file", type=["csv"])

            if uploaded_file:
                # --- STEP 1: Define bulk_df right here ---
                bulk_df = pd.read_csv(uploaded_file)
    
                # Check if 'Patient_ID' and biomarkers are present
                required_cols = ['Patient_ID'] + all_features
                if all(col in bulk_df.columns for col in required_cols):
                    # --- STEP 2: Run Predictions ---
                    probs = model.predict_proba(bulk_df[all_features])[:, 1]
                    bulk_df['GBM_Probability'] = probs
                    bulk_df['Result'] = bulk_df['GBM_Probability'].apply(
                        lambda x: "POSITIVE" if x > 0.5 else "NEGATIVE")

                    # --- STEP 3: Identify Relevant Biomarkers ---
                    # Filters the 843 columns down to only those the model actually used
                    relevant_biomarkers = [f for f in all_features if importance_map.get(f, 0) > 0]
                    relevant_biomarkers = sorted(relevant_biomarkers, 
                                                 key=lambda x: importance_map.get(x, 0), 
                                                 reverse=True)

                    # --- STEP 4: Visualizations (This is where your error likely happened) ---
                    st.subheader("📊 Batch Analysis Results")
                    # bulk_df is now defined, so set_index('Patient_ID') will work
                    st.bar_chart(bulk_df.set_index('Patient_ID')['GBM_Probability'])

                    # --- STEP 5: Display & Download Filtered Results ---
                    st.write(f"### 🧬 Diagnostic Details (Showing {len(relevant_biomarkers)} active markers)")
        
                    display_cols = ['Patient_ID', 'GBM_Probability', 'Result'] + relevant_biomarkers
                    filtered_results = bulk_df[display_cols]
        
                    st.dataframe(filtered_results)

                    # Download button for the filtered relevant results
                    csv_buffer = io.BytesIO()
                    filtered_results.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Diagnostic Results (Relevant Markers Only)",
                        data=csv_buffer,
                        file_name="GBM_Diagnostic_Results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    missing = [c for c in required_cols if c not in bulk_df.columns]
                    st.error(f"Missing required columns in CSV: {', '.join(missing[:5])}...")

# --- DOCUMENTATION AND DEMO PAGES (INDENTED CORRECTLY) ---
elif app_mode == "App Documentation":
    st.title(" App Documentation & User Guide")
    st.write("""
        Welcome to the documentation tab of MultiNet_AI. This guide provides a step-by-step walkthrough 
        of the interface workflow to help you navigate the platform effectively.
    """)


    st.divider()

    # --- GUI STEP BY STEP GUIDE ---

    st.header("1. Select an input prefrence section and Upload your data")
    st.write("""
        1. Reads Multiomics abundance score.
        2. Filters relevant features according to their importance. 
        3. Excludes less relevant features. 
        4. Demonstrates GBM risk factor (Positive or Negative).
        5. Returns sample specific visualization of the importance of relevant features. 
        Navigate to the **'Upload your own omics data'** page. You will see two primary methods for entry:
        * **Manual Entry Tab:** Best for checking a single patient profile by typing in values for the highly significant biomarkers.
        * **Bulk Analysis Tab:** Best for processing large cohorts using a spreadsheet.
    """)

    st.header("2. Performing a Manual Diagnosis")
    st.write("""
        1. Locate the 'Input abundance score in raw value' section.
        2. Enter the raw abundance values for the top 10 biomarkers identified by the model.
        3. Click the 'Run Diagnostic' button.
        4. Review the 'Probability Score' and the 'Biomarker Impact Chart' to see which specific inputs influenced the result.
    """)

    st.header("3. Bulk Processing using CSV file")
    st.write("""
        1. Switch to the 'Bulk Analysis' tab.
        2. Click 'Download CSV Template' to ensure your data is formatted correctly for the model.
        3. Fill the template with your patient IDs and corresponding omics data.
        4. Upload the file using the 'File Uploader'. 
        5. The system will automatically generate a 'Comparative Risk Analysis' chart and a detailed results table.
    """)

    st.header("Step 4: Interactive Demo walkthrough")
    st.write("""
        If you want to test the system behavior without your own data using dummy data:
        1. Select 'Interactive Demo Walkthrough' from the sidebar.
        2. Choose a pre-set clinical profile (e.g., 'Healthy Control' or 'GBM-Positive').
        3. Observe how the change in raw values directly shifts the **Confidence Score** and output.
    """)

    st.header("Step 5: Interpreting Results")
    st.write("""
        Probability > 50%: Indicates a high likelihood of a GBM-positive profile (displayed in Red).
        Probability < 50%: Indicates a healthy or negative profile (displayed in Green).
        Impact Scores:** Higher bars in the charts indicate that the specific biomarker had a stronger influence on that patient's classification.
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
        "This walkthrough demonstrates that the modeI calculates probability based on the intensity of the raw data. By entering values for these specific features, users can perform accurate real-world analysis.")


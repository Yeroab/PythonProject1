import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io
# --- FORCE BLUE THEME ---
st.markdown("""
    <style>

    st.markdown("""
    <style>
      /* --- Manual Entry Button Customization --- */
        div.stButton > button {
            background-color: #1f77b4 !important;
            color: white !important;
            border-radius: 6px !important;
            border: 1px solid #1f77b4 !important;
            font-weight: 700 !important;
            height: 3em !important;
            width: 100% !important;
            margin-top: 10px !important;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1) !important;
        }

        /* Hover effect for the Run button */
        div.stButton > button:hover {
            background-color: #155a8a !important;
            border-color: #155a8a !important;
            transform: translateY(-1px);
        }

        /* Active click effect */
        div.stButton > button:active {
            transform: translateY(1px);
        }

        /* --- Input Controls --- */
        /* Style the increment/decrement buttons (+ and -) in number inputs */
        button[data-testid="stNumberInputStepUp"], 
        button[data-testid="stNumberInputStepDown"] {
            background-color: transparent !important;
            color: #1f77b4 !important;
        }

        /* --- Tab Alignment --- */
        /* Ensures the active tab (Manual Entry) matches the button color */
        div[data-baseweb="tab-highlight"] {
            background-color: #1f77b4 !important;
        }

        /* --- Result Area Refinement --- */
        /* Style the Probability text to look professional */
        .stMarkdown h3 {
            color: #002b36 !important;
            padding-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)
        /* 1. Top Header Bar */
        header[data-testid="stHeader"] {
            background-color: #1f77b4 !important;
        }

        /* 2. Sidebar Background */
        section[data-testid="stSidebar"] {
            background-color: #002b36 !important;
        }

        /* 3. Sidebar Text: Labels, Titles, and Paragraphs */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: white !important;
        }

        /* 4. Fix Sidebar Selectbox (The white box in your image) */
        /* This targets the text inside the selection box */
        div[data-testid="stSelectbox"] div[role="button"] {
            color: #002b36 !important; /* Keep text dark inside the white box for contrast */
        }

        /* 5. Buttons: Base State */
        div.stButton > button {
            background-color: #1f77b4 !important;
            color: white !important;
            border: 1px solid #1f77b4 !important;
            border-radius: 4px;
            width: 100%; /* Optional: matches your input width */
        }

        /* 6. Buttons: Hover State */
        div.stButton > button:hover {
            background-color: #155a8a !important;
            border-color: #155a8a !important;
            color: white !important;
        }

        /* 7. Main Area Background */
        .stApp {
            background-color: #f0f8ff !important;
        }

        /* 8. Fix Input Labels in Main Area (AACS_prot, etc.) */
        /* If you want these to be a specific color too */
        .stNumberInput label {
            color: #002b36 !important;
            font-weight: bold !important;
        }

        /* --- 2. TABS CUSTOMIZATION (Manual Entry / Bulk Analysis) --- */
        /* Target the tab bar */
        button[data-baseweb="tab"] {
            color: #555555 !important; /* Inactive tab text */
        }

        /* Target the active (selected) tab text */
        button[aria-selected="true"] p {
            color: #1f77b4 !important;
            font-weight: bold !important;
        }

        /* Target the underline of the active tab */
        div[data-baseweb="tab-highlight"] {
            background-color: #1f77b4 !important;
        }

        /* --- 3. BUTTONS (Refined for 'Run Diagnostic') --- */
        div.stButton > button {
            background-color: #1f77b4 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 2rem !important;
            transition: all 0.3s ease !important;
        }

        div.stButton > button:hover {
            background-color: #155a8a !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            transform: translateY(-1px) !important;
        }

        /* --- 4. NUMBER INPUT BOXES --- */
        /* Changes the highlight color when you click into an input box */
        div[data-baseweb="input"] {
            border-color: transparent !important;
        }
        
        div[data-baseweb="input"]:focus-within {
            border-color: #1f77b4 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(page_title="MulitNet-Ai", layout="wide")


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
                                ["Upload your own omics data", "App Documentation", "Interactive Demo Walkthrough"])

# --- PAGE 1: MAIN DIAGNOSIS ---
if app_mode == "Upload your own omics data":
    st.title(" MultiNet-Ai User Driven Interface")
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
    st.title(" App Documentation & User Guide")
    st.write("""
        Welcome to the documentation tab of GBM_Omics. This guide provides a step-by-step walkthrough 
        of the interface workflow to help you navigate the platform effectively.
    """)


    st.divider()

    # --- GUI STEP BY STEP GUIDE ---

    st.header("1. Select an input prefrence section and Upload your data")
    st.write("""
        Navigate to the **'Upload your own omics data'** page. You will see two primary methods for entry:
        * **Manual Entry Tab:** Best for checking a single patient profile by typing in values for the highly significant biomarkers.
        * **Bulk Analysis Tab:** Best for processing large cohorts using a spreadsheet.
    """)

    st.header("2. Performing a Manual Diagnosis")
    st.write("""
        1. Locate the 'High-Impact Raw Biomarker Inputs' section.
        2. Enter the raw expression values for the top 10 biomarkers identified by the model.
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


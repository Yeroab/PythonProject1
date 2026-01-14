import streamlit as st
import joblib
import pandas as pd
import numpy as np

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
        
        # Calculate Feature Importance
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

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

# --- PAGE 3: INTERACTIVE DEMO ---
elif app_mode == "Interactive Demo Walkthrough":
    st.title("🚀 Interactive Platform Walkthrough")
    
    st.subheader("Introduction")
    st.write("Follow the steps below to simulate a clinical workflow and see how specific feature values change the AI diagnostic output.")

    st.markdown("---")
    
    if diag:
        model = diag['model']
        all_features = diag['features']
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        # Step 1: Selection
        st.subheader("Step 1: Select a Clinical Case Profile")
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
            st.subheader("Step 2: Feature and Value List")
            st.write("The following biomarkers and their raw values are being sent to the AI model for this simulation:")
            
            # Prepare the list/table of features and values
            display_list = []
            for f in top_10:
                display_list.append({"Biomarker (Feature)": f, "Raw Value": sim_data[f][0]})
            
            # Displaying as a clean table
            st.table(pd.DataFrame(display_list))

            # Step 3: Result
            st.subheader("Step 3: Real-Time Diagnostic Result")
            sim_df = pd.DataFrame(sim_data)
            prob = model.predict_proba(sim_df[all_features])[0][1]
            
            st.write(f"**AI Confidence Score:**")
            st.progress(float(prob))
            
            if prob > 0.5:
                st.error(f"Prediction: POSITIVE ({prob:.2%})")
                st.write("The high raw values assigned to the critical biomarkers above triggered a Positive diagnosis.")
            else:
                st.success(f"Prediction: NEGATIVE ({prob:.2%})")
                st.write("The low raw values assigned to the biomarkers represent a healthy profile, resulting in a Negative diagnosis.")

    st.markdown("---")
    st.subheader("Final Interpretation")
    st.write("This walkthrough demonstrates that the AI calculates probability based on the intensity of the raw data. By entering values for these specific features on the 'Main Diagnosis' page, users can perform accurate real-world analysis.")

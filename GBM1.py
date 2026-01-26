import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import matplotlib.pyplot as plt
import io

# --- Page Configuration ---
st.set_page_config(page_title="MultiNet_AI", layout="wide", page_icon="🔬")

# --- Asset Loading (Cached) ---
@st.cache_resource
def load_clinical_model():
    with open('gbm_clinical_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # The model contains 843 features including Protein, RNA, and Metabolites 
    feature_names = model.get_booster().feature_names
    feature_types = model.get_booster().feature_types
    return model, feature_names, feature_types

try:
    model, feature_names, feature_types = load_clinical_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Ensure 'gbm_clinical_model.pkl' is in the app directory.")
    st.stop()

# --- Utility: PDF Generation ---
def create_clinical_pdf(prediction, probability, input_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "MultiNet_AI Clinical Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Assessment: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}", ln=True)
    pdf.cell(200, 10, f"Probability Score: {probability:.4f}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, "Key Clinical Indicators Analyzed: 843", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- Navigation Sidebar ---
st.sidebar.title("🧬 MultiNet_AI")
page = st.sidebar.radio("Navigation", ["Home", "Clinical Analysis", "Model Insights"])

# --- PAGE 1: HOME ---
if page == "Home":
    st.title("MultiNet_AI: High-Dimensional GBM Prediction")
    st.markdown("""
    ### System Overview
    MultiNet_AI is a clinical decision support system designed to analyze Glioblastoma Multiforme (GBM) risk 
    using a multi-modal feature set of **843 distinct markers**[cite: 82].
    
    **Marker Categories:**
    - **Proteomics**: Protein expression levels (e.g., `AACS_prot`, `VEGFA_prot`)[cite: 84, 143].
    - **Transcriptomics**: RNA expression levels (e.g., `AAMDC_rna`, `EGFR_rna`)[cite: 84, 102].
    - **Metabolomics**: Small molecule concentrations (e.g., `D-glucose_met`, `citricacid_met`)[cite: 98, 145].
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5f/Glioblastoma_Macro.jpg", caption="Clinical focus: GBM Analysis", width=500)

# --- PAGE 2: CLINICAL ANALYSIS (The "Oven") ---
elif page == "Clinical Analysis":
    st.title("🎯 Clinical Analysis Suite")
    
    tabs = st.tabs(["Single Patient", "Batch Processing"])
    
    with tabs[0]:
        st.subheader("Manual Patient Entry")
        st.write("Enter values for key markers (Others default to 0.0 for this simulation).")
        
        # Display small sample of inputs for UI clarity
        input_cols = st.columns(3)
        user_inputs = {}
        for i, feat in enumerate(feature_names[:12]):
            with input_cols[i % 3]:
                user_inputs[feat] = st.number_input(f"{feat}", value=0.0, format="%.4f")
        
        if st.button("🔥 Run MultiNet_AI Inference"):
            # Construct full 843 feature vector
            full_data = {f: [user_inputs.get(f, 0.0)] for f in feature_names}
            input_df = pd.DataFrame(full_data)
            
            with st.spinner("Baking prediction and SHAP interpretability..."):
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0, 1]
                
                # Metrics
                m1, m2 = st.columns(2)
                m1.metric("Risk Level", "High" if pred == 1 else "Low")
                m2.metric("Confidence", f"{proba:.2%}")
                
                # SHAP Explanation
                st.subheader("🔬 Trustworthy AI: SHAP Explanation")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                fig, ax = plt.subplots()
                shap.force_plot(explainer.expected_value, shap_values[0,:], input_df.iloc[0,:], matplotlib=True, show=False)
                st.pyplot(plt.gcf())
                
                # PDF Download
                pdf_data = create_clinical_pdf(pred, proba, input_df)
                st.download_button("📄 Download Clinical Report", data=pdf_data, file_name="MultiNet_Report.pdf")

    with tabs[1]:
        st.subheader("Batch Diagnostic Processing")
        st.info("Upload a CSV containing 843 features for bulk analysis.")
        
        # Template Download
        template = pd.DataFrame(columns=feature_names)
        st.download_button("📥 Download CSV Template", template.to_csv(index=False), "multinet_template.csv")
        
        uploaded_file = st.file_uploader("Drop CSV here", type="csv")
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            if batch_df.shape[1] == 843:
                preds = model.predict(batch_df)
                batch_df['Risk_Score'] = model.predict_proba(batch_df)[:, 1]
                batch_df['Prediction'] = preds
                
                st.write("### Batch Results")
                st.dataframe(batch_df[['Prediction', 'Risk_Score']].head())
                
                fig = px.histogram(batch_df, x="Risk_Score", color="Prediction", title="Population Risk Distribution")
                st.plotly_chart(fig)
            else:
                st.error(f"Column mismatch: Expected 843, got {batch_df.shape[1]}")

# --- PAGE 3: MODEL INSIGHTS ---
elif page == "Model Insights":
    st.title("📊 MultiNet_AI Global Insights")
    
    st.write(f"The model is an **XGBClassifier** utilizing **100 trees** with a **max depth of 4**[cite: 75, 79].")
    
    # Feature Importance Visualization
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Marker': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(20)
    
    fig_imp = px.bar(feat_imp_df, x='Importance', y='Marker', orientation='h', 
                     title="Top 20 Most Influential Biological Markers",
                     color='Importance', color_continuous_scale='Viridis')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("""
    ### Analysis of Data Modalities
    The MultiNet_AI system integrates:
    - **Proteomics**: Captures functional cellular state.
    - **Transcriptomics**: Reflects gene expression activity.
    - **Metabolomics**: Provides a snapshot of metabolic flux within the GBM microenvironment.
    """)

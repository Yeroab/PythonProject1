import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io

# 1. Configuration & Model Loading [cite: 1, 2]
st.set_page_config(page_title="GBM Clinical Predictor", layout="wide")

@st.cache_resource
def load_assets():
    with open('gbm_clinical_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Extract feature names and types from the model booster 
    feature_names = model.get_booster().feature_names
    feature_types = model.get_booster().feature_types
    return model, feature_names, feature_types

model, feature_names, feature_types = load_assets()

# 2. Sidebar - Navigation & Template Download
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Go to", ["Single Prediction", "Batch Processing", "Model Insights"])

# Template Download [cite: 3]
template_df = pd.DataFrame(columns=feature_names)
csv_template = template_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download CSV Template", data=csv_template, file_name="gbm_template.csv", mime="text/csv")

# 3. Core Functions
def generate_pdf_report(prediction, proba, shap_values, features):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "GBM Clinical Prediction Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Prediction Result: {'High Risk' if prediction == 1 else 'Low Risk'}", ln=True)
    pdf.cell(200, 10, f"Confidence: {proba[1]:.2%}", ln=True)
    # Add more clinical notes here
    return pdf.output(dest='S').encode('latin-1')

# 4. Main App Logic
if app_mode == "Batch Processing":
    st.header("📊 Batch Clinical Processing")
    uploaded_file = st.file_uploader("Drag and drop clinical data CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if len(df.columns) == 843:
            # Batch Prediction [cite: 2]
            preds = model.predict(df)
            probs = model.predict_proba(df)[:, 1]
            df['Prediction'] = preds
            df['Probability'] = probs
            
            st.write("### Prediction Results")
            st.dataframe(df[['Prediction', 'Probability']].head())
            
            # Interactive Dashboard
            fig = px.histogram(df, x="Probability", color="Prediction", 
                               title="Distribution of Risk Probabilities",
                               color_discrete_map={0: "green", 1: "red"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Invalid format. Expected 843 features, found {len(df.columns)}.")

elif app_mode == "Single Prediction":
    st.header("🎯 Clinical Sample Prediction")
    st.info("Input sample values to generate a risk assessment and SHAP explanation.")
    
    # Create input fields for key features (Top 10 for brevity)
    inputs = {}
    cols = st.columns(4)
    for i, feat in enumerate(feature_names[:12]):
        with cols[i % 4]:
            inputs[feat] = st.number_input(feat, value=0.0)
    
    # Fill remaining 843 features with 0 for the demo
    full_input = {f: [inputs.get(f, 0.0)] for f in feature_names}
    input_df = pd.DataFrame(full_input)

    if st.button("Generate Prediction & SHAP Report"):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        
        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Classification", "High" if pred == 1 else "Low")
            st.metric("Confidence Score", f"{proba[1]:.2%}")
        
        # 🔥 SHAP Explanations
        with col2:
            st.subheader("Interpretable SHAP Summary")
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            
            # Matplotlib to Streamlit
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.force_plot(explainer.expected_value, shap_vals[0,:], input_df.iloc[0,:], matplotlib=True)
            st.pyplot(bbox_inches='tight')

        # 📄 Automated PDF Report
        pdf_bytes = generate_pdf_report(pred, proba, shap_vals, feature_names)
        st.download_button("Download Clinical PDF Report", data=pdf_bytes, file_name="clinical_report.pdf")

elif app_mode == "Model Insights":
    st.header("📈 Model Performance & Architecture")
    st.write(f"The model is an **XGBClassifier** utilizing **{model.n_estimators} trees** with a **max depth of 4**[cite: 1, 2, 79].")
    
    # Global Feature Importance
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(20)
    
    fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', title="Top 20 Predictive Features")
    st.plotly_chart(fig_imp, use_container_width=True)

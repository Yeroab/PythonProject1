import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="MultiNet_AI", layout="wide", page_icon="🧬")

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    with open('gbm_clinical_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names = model.get_booster().feature_names
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Biomarker': feature_names,
        'Influence Score': importances
    }).sort_values(by='Influence Score', ascending=False)
    
    return model, feature_names, importance_df

try:
    model, feature_names, importance_df = load_assets()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- Section: Processing Engine ---
def process_data(df):
    df_aligned = df.reindex(columns=feature_names, fill_value=0.0)
    probs = model.predict_proba(df_aligned.astype(float))[:, 1]
    preds = (probs > 0.5).astype(int)
    
    results = pd.DataFrame({
        "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
        "Risk Score": probs
    })
    return pd.concat([results, df_aligned.reset_index(drop=True)], axis=1)

# --- Section: Prediction & Risk Charts ---
def render_risk_charts(results, mode="manual", key_prefix=""):
    st.subheader("🎯 Prediction & Risk Assessment")
    
    if mode == "manual":
        # Individual Gauge Chart for Single Patient
        prob = results["Risk Score"].iloc[0]
        pred = results["Prediction"].iloc[0]
        color = "#EF553B" if pred == "High Risk" else "#00CC96"
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Patient Risk Assessment: {pred}", 'font': {'size': 24, 'color': color}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"{key_prefix}_gauge")

    else:
        # Pie Chart and Histogram for Bulk Processing
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(results, names='Prediction', title="Cohort Classification Summary",
                             color='Prediction', color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#636EFA"})
            st.plotly_chart(fig_pie, use_container_width=True, key=f"{key_prefix}_pie")
        with c2:
            fig_hist = px.histogram(results, x="Risk Score", color="Prediction",
                                     title="Risk Score Probability Distribution",
                                     color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#636EFA"})
            st.plotly_chart(fig_hist, use_container_width=True, key=f"{key_prefix}_hist")

# --- Section: Individual Dashboard ---
def render_dashboard(results, mode="manual", key_prefix=""):
    # 1. Prediction Charts (New Section)
    render_risk_charts(results, mode=mode, key_prefix=key_prefix)
    
    # 2. Global Influence (The Probability List of Biomarkers)
    st.divider()
    st.subheader("🧬 Global Biomarker Influence")
    fig_imp = px.bar(importance_df.head(15), x='Influence Score', y='Biomarker', 
                     orientation='h', color='Influence Score', color_continuous_scale='Reds')
    st.plotly_chart(fig_imp, use_container_width=True, key=f"{key_prefix}_imp")

    # 3. Individual Explorer (For Deep Dives)
    st.divider()
    st.subheader("🔍 Individual Patient Deep-Dive")
    selected_idx = st.selectbox("Select Record", results.index, key=f"{key_prefix}_select")
    patient_row = results.iloc[selected_idx]
    
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.write("### Biological Signature")
        prot_avg = patient_row.filter(like='_prot').mean()
        rna_avg = patient_row.filter(like='_rna').mean()
        met_avg = patient_row.filter(like='_met').mean()
        
        fig_radar = go.Figure(data=go.Scatterpolar(r=[prot_avg, rna_avg, met_avg],
            theta=['Proteins', 'RNA', 'Metabolites'], fill='toself'))
        st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}_radar")

    with col_r:
        st.write(f"### Top Raw Markers for Patient {selected_idx}")
        markers = patient_row.drop(['Prediction', 'Risk Score'])
        top_20 = markers.astype(float).sort_values(ascending=False).head(20)
        fig_bar = px.bar(x=top_20.values, y=top_20.index, orientation='h', color_continuous_scale='Viridis')
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_pbar")

# --- MAIN UI ---
st.title("🧬 MultiNet_AI | Clinical Diagnostic Suite")
t1, t2 = st.tabs(["✍️ Manual Entry", "💾 Bulk Upload"])

with t1:
    st.header("✍️ Manual Patient Entry")
    user_inputs = {name: st.number_input(f"{name}", value=0.0, key=f"m_{name}") for name in feature_names[:12]}
    # (Advanced markers can be added in an expander as in previous turns)
    if st.button("Run Manual Analysis"):
        m_results = process_data(pd.DataFrame([user_inputs]))
        render_dashboard(m_results, mode="manual", key_prefix="man")

with t2:
    st.header("💾 Bulk Data Processing")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        b_results = process_data(pd.read_csv(uploaded_file))
        render_dashboard(b_results, mode="bulk", key_prefix="blk")

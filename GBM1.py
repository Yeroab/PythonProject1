import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io

# --- Asset Loading ---
@st.cache_resource
def load_bundle():
    with open('gbm_clinical_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names = model.get_booster().feature_names
    return model, feature_names

model, feature_names = load_bundle()

# --- Section: 💾 PROCESSING ENGINE ---
def process_data(df):
    df_aligned = df.reindex(columns=feature_names, fill_value=0.0)
    with st.spinner("Calculating Risk Probabilities..."):
        probs = model.predict_proba(df_aligned.astype(float))[:, 1]
        preds = (probs > 0.5).astype(int)
        
        results = pd.DataFrame({
            "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
            "Risk Score": probs
        })
        return pd.concat([results, df_aligned.reset_index(drop=True)], axis=1)

# --- Section: 📊 DASHBOARD WITH RISK PROBABILITY LIST ---
def render_dashboard(results):
    st.header("📊 MultiNet Diagnostic Insights")
    
    # 1. Global Biomarker Risk Influence (Feature Importance)
    st.subheader("🧬 Top Biomarker Risk Influence")
    st.write("These markers have the highest statistical impact on the model's risk calculation.")
    
    # Extracting Global Importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Biomarker': feature_names,
        'Influence Score': importances
    }).sort_values(by='Influence Score', ascending=False)

    # Display Top 15 as a Bar Chart
    fig_imp = px.bar(
        importance_df.head(15), 
        x='Influence Score', y='Biomarker', 
        orientation='h',
        color='Influence Score',
        color_continuous_scale='Reds',
        title="Top 15 Biomarkers Driving High Risk"
    )
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

    # 2. Complete Probability List (Searchable Table)
    with st.expander("📄 View Full Biomarker Influence List (843 Markers)"):
        st.dataframe(importance_df, use_container_width=True, hide_index=True)

    st.divider()

    # 3. Individual Patient Analysis
    st.subheader("🔍 Individual Patient Explorer")
    selected_idx = st.selectbox("Select Patient Row", results.index)
    patient_row = results.iloc[selected_idx]

    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.write("### Biological Signature")
        prot_avg = patient_row.filter(like='_prot').mean()
        rna_avg = patient_row.filter(like='_rna').mean()
        met_avg = patient_row.filter(like='_met').mean()
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[prot_avg, rna_avg, met_avg],
            theta=['Proteins', 'RNA', 'Metabolites'],
            fill='toself'
        ))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r:
        st.write(f"### Top Raw Levels for Patient {selected_idx}")
        markers = patient_row.drop(['Prediction', 'Risk Score'])
        top_20 = markers.astype(float).sort_values(ascending=False).head(20)
        
        fig_bar = px.bar(
            x=top_20.values, y=top_20.index, orientation='h',
            color=top_20.values, color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --- MAIN APP UI ---
st.title("🧬 MultiNet_AI | Clinical Diagnostic Suite")
tab_manual, tab_batch = st.tabs(["✍️ Single Entry", "💾 Batch Processing"])

with tab_manual:
    # (Assuming render_manual_entry code from previous turn is here)
    pass 

with tab_batch:
    st.header("💾 Batch Data Processing")
    uploaded_file = st.file_uploader("📂 Upload Raw Clinical Data (CSV)", type=["csv"])
    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        results = process_data(input_data)
        render_dashboard(results)

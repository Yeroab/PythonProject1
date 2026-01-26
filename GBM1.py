import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io

# --- Page Configuration ---
st.set_page_config(page_title="MultiNet_AI", layout="wide", page_icon="🧬")

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    with open('gbm_clinical_model.pkl', 'rb') as f:
        # Load the dictionary and extract the model
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names = model.get_booster().feature_names
    
    # Calculate Global Feature Importance (Influence)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Biomarker': feature_names,
        'Influence Score': importances
    }).sort_values(by='Influence Score', ascending=False)
    
    return model, feature_names, importance_df

try:
    model, feature_names, importance_df = load_assets()
except Exception as e:
    st.error(f"Initialization Error: {e}. Check if 'gbm_clinical_model.pkl' is present.")
    st.stop()

# --- Section: Processing Engine ---
def process_data(df):
    # Alignment: Ensure columns match the 843 markers exactly
    df_aligned = df.reindex(columns=feature_names, fill_value=0.0)
    
    with st.spinner("Analyzing Raw Biomarkers..."):
        # Inference on raw values (no Z-scaling)
        probs = model.predict_proba(df_aligned.astype(float))[:, 1]
        preds = (probs > 0.5).astype(int)
        
        results = pd.DataFrame({
            "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
            "Risk Score": probs
        })
        # Merge results with input data
        return pd.concat([results, df_aligned.reset_index(drop=True)], axis=1)

# --- Section: Manual Entry UI ---
def render_manual_entry():
    st.header("✍️ Manual Patient Entry")
    st.info("Input raw values. Markers left at 0.0 are treated as baseline.")
    
    user_inputs = {}
    # Top 12 markers for quick access
    top_cols = st.columns(3)
    for i, name in enumerate(feature_names[:12]):
        with top_cols[i % 3]:
            user_inputs[name] = st.number_input(f"{name}", value=0.0)

    with st.expander("🛠️ Advanced Marker Settings (Full 843 Set)"):
        adv_cols = st.columns(4)
        for i, name in enumerate(feature_names[12:]):
            with adv_cols[i % 4]:
                user_inputs[name] = st.number_input(f"{name}", value=0.0, key=f"man_{name}")

    if st.button("🚀 Run Single Patient Analysis"):
        return pd.DataFrame([user_inputs])
    return None

# --- Section: Interactive Dashboard ---
def render_dashboard(results):
    st.header("📊 MultiNet Diagnostic Insights")
    
    # 1. Influence Leaderboard
    st.subheader("🧬 Global Biomarker Risk Influence")
    fig_imp = px.bar(
        importance_df.head(15), 
        x='Influence Score', y='Biomarker', 
        orientation='h',
        color='Influence Score', color_continuous_scale='Reds',
        title="Top 15 Biomarkers Driving the Model"
    )
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("📄 Full Biomarker Probability List (843 Markers)"):
        st.dataframe(importance_df, use_container_width=True, hide_index=True)

    st.divider()

    # 2. Population Risk Distribution
    st.subheader("📈 Cohort Risk Analysis")
    fig_hist = px.histogram(
        results, x="Risk Score", color="Prediction",
        title="Distribution of Calculated Risk Scores",
        color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#636EFA"},
        nbins=30, marginal="rug"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # 3. Individual Patient Explorer
    st.subheader("🔍 Individual Patient Deep-Dive")
    selected_idx = st.selectbox("Select Patient Record", results.index)
    patient_row = results.iloc[selected_idx]

    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.write("### Biological Signature")
        # Aggregating by data type
        prot_avg = patient_row.filter(like='_prot').mean()
        rna_avg = patient_row.filter(like='_rna').mean()
        met_avg = patient_row.filter(like='_met').mean()
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[prot_avg, rna_avg, met_avg],
            theta=['Proteins (_prot)', 'RNA (_rna)', 'Metabolites (_met)'],
            fill='toself'
        ))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r:
        st.write(f"### Top Raw Levels for Patient {selected_idx}")
        # Drop metadata for bar chart
        markers = patient_row.drop(['Prediction', 'Risk Score'])
        top_20 = markers.astype(float).sort_values(ascending=False).head(20)
        
        fig_bar = px.bar(
            x=top_20.values, y=top_20.index, orientation='h',
            labels={'x': 'Raw Value', 'y': 'Marker'},
            color=top_20.values, color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --- Main App Logic ---
st.title("🧬 MultiNet_AI | GBM Clinical Intelligence")

tab_manual, tab_batch = st.tabs(["✍️ Single Patient Entry", "💾 Batch Upload"])

with tab_manual:
    manual_data = render_manual_entry()
    if manual_data is not None:
        results = process_data(manual_data)
        render_dashboard(results)

with tab_batch:
    st.header("💾 Batch Data Processing")
    col_u1, col_u2 = st.columns([2,1])
    with col_u2:
        # Template Download
        template = pd.DataFrame(columns=feature_names)
        st.download_button("📥 Download Template", template.to_csv(index=False), "template.csv")
    
    with col_u1:
        uploaded_file = st.file_uploader("Upload CSV (843 markers)", type="csv")
    
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        results = process_data(input_df)
        render_dashboard(results)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io

# --- Page Configuration ---
st.set_page_config(page_title="MultiNet_AI", layout="wide", page_icon="🧬")

# --- Asset Loading (Model & Feature Metadata) ---
@st.cache_resource
def load_assets():
    try:
        with open('gbm_clinical_model.pkl', 'rb') as f:
            bundle = pickle.load(f)
        
        # Extract model from dictionary
        model = bundle["model"]
        feature_names = model.get_booster().feature_names
        
        # Calculate Global Feature Importance (Risk Probability Influence)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Biomarker': feature_names,
            'Influence Score': importances
        }).sort_values(by='Influence Score', ascending=False)
        
        return model, feature_names, importance_df
    except FileNotFoundError:
        st.error("⚠️ File 'gbm_clinical_model.pkl' not found. Please ensure it is in the root directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Initialization Error: {e}")
        st.stop()

model, feature_names, importance_df = load_assets()

# --- Section: Processing Engine (Direct Raw Values) ---
def process_data(df):
    # Align user input with the 843 markers expected by the model
    df_aligned = df.reindex(columns=feature_names, fill_value=0.0)
    
    with st.spinner("Analyzing Patient Biomarkers..."):
        # Inference using raw values (as requested)
        probs = model.predict_proba(df_aligned.astype(float))[:, 1]
        preds = (probs > 0.5).astype(int)
        
        results = pd.DataFrame({
            "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
            "Risk Score": probs
        })
        # Merge risk results with original marker data
        return pd.concat([results, df_aligned.reset_index(drop=True)], axis=1)

# --- Section: Risk & Prediction Visuals ---
def render_risk_charts(results, mode="manual", key_prefix=""):
    st.subheader("🎯 Prediction & Risk Assessment")
    
    if mode == "manual":
        # Individual Gauge for Single Entry
        prob = results["Risk Score"].iloc[0]
        pred = results["Prediction"].iloc[0]
        color = "#EF553B" if pred == "High Risk" else "#00CC96"
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Assessment: {pred}", 'font': {'size': 24, 'color': color}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}]}))
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"{key_prefix}_gauge")
    else:
        # Pie & Histogram for Bulk Entry
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(results, names='Prediction', title="Cohort Summary",
                             color='Prediction', color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#636EFA"})
            st.plotly_chart(fig_pie, use_container_width=True, key=f"{key_prefix}_pie")
        with c2:
            fig_hist = px.histogram(results, x="Risk Score", color="Prediction",
                                     title="Risk Probability Distribution",
                                     color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#636EFA"})
            st.plotly_chart(fig_hist, use_container_width=True, key=f"{key_prefix}_hist")

# --- Section: Complete Dashboard ---
def render_dashboard(results, mode="manual", key_prefix=""):
    # 1. Prediction Visuals
    render_risk_charts(results, mode=mode, key_prefix=key_prefix)
    
    # 2. Global Risk Influence (Probability List of Biomarkers)
    st.divider()
    st.subheader("🧬 Global Biomarker Influence")
    st.write("Top markers driving the risk probability across the model.")
    fig_imp = px.bar(importance_df.head(15), x='Influence Score', y='Biomarker', 
                     orientation='h', color='Influence Score', color_continuous_scale='Reds')
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True, key=f"{key_prefix}_imp")

    with st.expander("📄 View Searchable Influence List (All 843 Markers)"):
        st.dataframe(importance_df, use_container_width=True, hide_index=True)

    # 3. Individual Patient Explorer
    st.divider()
    st.subheader("🔍 Individual Patient Deep-Dive")
    selected_idx = st.selectbox("Select Patient Record", results.index, key=f"{key_prefix}_select")
    patient_row = results.iloc[selected_idx]
    
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.write("### Multi-Modal Signature")
        # Group by marker suffix
        prot_avg = patient_row.filter(like='_prot').mean()
        rna_avg = patient_row.filter(like='_rna').mean()
        met_avg = patient_row.filter(like='_met').mean()
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[prot_avg, rna_avg, met_avg],
            theta=['Proteins', 'RNA', 'Metabolites'], fill='toself'))
        st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}_radar")

    with col_r:
        st.write(f"### Top 20 Raw Marker Levels (Patient {selected_idx})")
        markers = patient_row.drop(['Prediction', 'Risk Score'])
        top_20 = markers.astype(float).sort_values(ascending=False).head(20)
        fig_bar = px.bar(x=top_20.values, y=top_20.index, orientation='h', 
                         color=top_20.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_pbar")

# --- MAIN INTERFACE ---
st.title("🧬 MultiNet_AI | GBM Clinical Diagnostic Suite")

tab_manual, tab_batch = st.tabs(["✍️ Manual Patient Entry", "💾 Bulk Data Upload"])

with tab_manual:
    st.header("✍️ Manual Patient Entry")
    st.info("Input raw laboratory values. Markers left at 0.0 will be treated as baseline.")
    
    # Manual Entry Fields
    user_inputs = {}
    m_cols = st.columns(3)
    # High-influence markers first
    for i, name in enumerate(feature_names[:12]):
        with m_cols[i % 3]:
            user_inputs[name] = st.number_input(f"{name}", value=0.0, key=f"man_in_{name}")
            
    with st.expander("🛠️ Advanced Marker Input (Full 843 Set)"):
        adv_cols = st.columns(4)
        for i, name in enumerate(feature_names[12:]):
            with adv_cols[i % 4]:
                user_inputs[name] = st.number_input(f"{name}", value=0.0, key=f"man_adv_{name}")

    if st.button("🚀 Analyze Single Patient", key="btn_manual"):
        m_results = process_data(pd.DataFrame([user_inputs]))
        render_dashboard(m_results, mode="manual", key_prefix="man")

with tab_batch:
    st.header("💾 Bulk Data Processing")
    
    # Template Generation & Download
    col_t1, col_t2 = st.columns([2, 1])
    with col_t2:
        st.write("### ⬇️ Download Template")
        # Generate empty template with 843 columns
        template_csv = pd.DataFrame(columns=feature_names).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Template",
            data=template_csv,
            file_name="MultiNet_Patient_Template.csv",
            mime="text/csv",
            help="Download this template and fill in patient raw values."
        )
    
    with col_t1:
        st.write("### ⬆️ Upload Patient Data")
        uploaded_file = st.file_uploader("Upload filled MultiNet CSV Template", type="csv")
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        # Process and show dashboard
        b_results = process_data(raw_df)
        render_dashboard(b_results, mode="bulk", key_prefix="blk")

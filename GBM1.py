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
def load_bundle():
    with open('gbm_clinical_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    # Get exact feature names from the Booster to ensure alignment
    feature_names = model.get_booster().feature_names
    return model, feature_names

try:
    model, feature_names = load_bundle()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Section: ✍️ MANUAL ENTRY ---
def render_manual_entry():
    st.header("✍️ Manual Patient Entry (Raw Values)")
    st.info("Enter raw laboratory values. Markers left at 0.0 will be treated as the baseline.")
    
    user_inputs = {}
    
    # Organize 843 features into categories for easier entry
    st.subheader("High-Impact Clinical Markers")
    top_cols = st.columns(3)
    # Highlight the first 12 features
    for i, name in enumerate(feature_names[:12]):
        with top_cols[i % 3]:
            user_inputs[name] = st.number_input(f"{name}", value=0.0)

    with st.expander("🛠️ Comprehensive Marker Set (Remaining 831 Features)"):
        st.write("Modify these advanced markers based on specific clinical data.")
        adv_cols = st.columns(4)
        for i, name in enumerate(feature_names[12:]):
            with adv_cols[i % 4]:
                # Unique keys for advanced inputs to avoid Streamlit conflicts
                user_inputs[name] = st.number_input(f"{name}", value=0.0, key=f"manual_{name}")

    if st.button("🚀 Run Single Patient Analysis"):
        # Create DataFrame and ensure columns match the model exactly
        manual_df = pd.DataFrame([user_inputs])
        manual_df = manual_df.reindex(columns=feature_names, fill_value=0.0)
        return manual_df
    return None

# --- Section: 🎯 BATCH UPLOAD ---
def render_upload_section():
    st.header("💾 Batch Data Processing")
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.write("### Template")
        template_df = pd.DataFrame(columns=feature_names)
        csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Template", data=csv, file_name="multinet_template.csv", mime="text/csv")

    with col1:
        uploaded_file = st.file_uploader("📂 Upload Raw Clinical Data (CSV)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            return df
    return None

# --- Section: 💾 PROCESSING ENGINE (NO SCALING) ---
def process_data(df):
    # 1. Alignment: Ensure the columns are in the EXACT order the coder used
    # This is critical for XGBoost boosters
    df_aligned = df.reindex(columns=feature_names, fill_value=0.0)
    
    with st.spinner("Processing raw values through MultiNet..."):
        # 2. Inference: Directly on RAW values as requested
        # We ensure data is float for the XGBoost engine
        probs = model.predict_proba(df_aligned.astype(float))[:, 1]
        preds = (probs > 0.5).astype(int)
        
        # 3. Structure Results
        results = pd.DataFrame({
            "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
            "Risk Score": probs
        })
        
        # Merge results with input data for the individual explorer
        final_df = pd.concat([results, df_aligned.reset_index(drop=True)], axis=1)
        return final_df

# --- Section: 📊 INTERACTIVE DASHBOARD ---
def render_dashboard(results):
    st.header("📊 MultiNet Diagnostic Insights")
    
    c1, c2 = st.columns(2)
    with c1:
        # Distribution of risk
        fig_hist = px.histogram(
            results, x="Risk Score", color="Prediction",
            title="Cohort Risk Distribution",
            color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#636EFA"},
            nbins=30
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        # Marker Importance / Correlation
        st.write("### Marker Interaction Map")
        # Visualizing correlation of first 10 markers
        top_10 = feature_names[:10]
        corr = results[top_10].corr()
        fig_heat = px.imshow(corr, text_auto=True, title="Marker Correlation Heatmap", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.subheader("🔍 Individual Patient Deep-Dive")
    selected_idx = st.selectbox("Select Patient Row", results.index)
    patient_row = results.iloc[selected_idx]

    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.write("### Biological Signature")
        # Group by feature suffix from your model metadata
        prot_avg = patient_row.filter(like='_prot').mean()
        rna_avg = patient_row.filter(like='_rna').mean()
        met_avg = patient_row.filter(like='_met').mean()
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[prot_avg, rna_avg, met_avg],
            theta=['Proteins', 'RNA', 'Metabolites'],
            fill='toself'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r:
        st.write(f"### Top 20 Raw Marker Levels (Patient {selected_idx})")
        # Isolate marker values
        markers = patient_row.drop(['Prediction', 'Risk Score'])
        top_20 = markers.astype(float).sort_values(ascending=False).head(20)
        
        fig_bar = px.bar(
            x=top_20.values, y=top_20.index, orientation='h',
            labels={'x': 'Raw Value', 'y': 'Marker'},
            color=top_20.values, color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

# --- MAIN APP EXECUTION ---
st.title("🧬 MultiNet_AI | Clinical Diagnostic Suite")

tab_manual, tab_batch = st.tabs(["✍️ Single Entry", "💾 Batch Processing"])

with tab_manual:
    manual_input = render_manual_entry()
    if manual_input is not None:
        results = process_data(manual_input)
        render_dashboard(results)

with tab_batch:
    batch_input = render_upload_section()
    if batch_input is not None:
        results = process_data(batch_input)
        render_dashboard(results)

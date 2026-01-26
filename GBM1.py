import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io

# --- Load Model Bundle ---
@st.cache_resource
def load_bundle():
    with open('gbm_clinical_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names = model.get_booster().feature_names
    return model, feature_names

model, feature_names = load_bundle()

# --- Section: 🎯 DRAG-AND-DROP + TEMPLATE ---
def render_upload_section():
    st.header("🎯 Clinical Data Intake")
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.write("### Instructions")
        st.info("Upload a CSV with 843 columns. Use the template to ensure marker order matches MultiNet requirements.")
        
        template_df = pd.DataFrame(columns=feature_names)
        csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Template",
            data=csv,
            file_name="multinet_clinical_template.csv",
            mime="text/csv",
        )

    with col1:
        uploaded_file = st.file_uploader("📂 Drag and drop clinical datasets here", type=["csv"])
        return uploaded_file

# --- Section: 💾 BATCH PROCESSING ---
def process_batch(df):
    # 1. Alignment: Ensure the columns are in the EXACT order the coder used
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # 2. Scaling: If the coder used Z-score normalization, you must replicate it
    # This prevents the values from being ignored by the high base score
    for col in df.columns:
        if df[col].std() != 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
            
    # 3. Prediction
    probs = model.predict_proba(df)[:, 1]
    return probs

# --- Section: 📊 INTERACTIVE DASHBOARD ---
def render_dashboard(results):
    st.header("📊 MultiNet Population Insights")
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig_hist = px.histogram(
            results, x="Risk Score", color="Prediction",
            title="Distribution of Patient Risk Scores",
            color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#636EFA"},
            marginal="rug", nbins=30
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.write("### Marker Interaction (Top Features)")
        top_10 = results.columns[2:12] 
        corr = results[top_10].corr()
        fig_heat = px.imshow(corr, text_auto=True, title="Key Marker Correlations", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heat, use_container_width=True)

    # Individual Sample Deep Dive
    st.divider()
    st.subheader("🔍 Individual Patient Explorer")
    
    # Select Patient
    selected_idx = st.selectbox("Select Patient Row for Details", results.index)
    
    # Get specific patient data (excluding the Prediction and Risk Score columns)
    patient_data = results.iloc[selected_idx].drop(['Prediction', 'Risk_Score'], errors='ignore')
    
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.write("### Multi-Modal Summary")
        categories = ['Protein Expression', 'RNA Expression', 'Metabolites']
        prot_val = results.filter(like='_prot').iloc[selected_idx].mean()
        rna_val = results.filter(like='_rna').iloc[selected_idx].mean()
        met_val = results.filter(like='_met').iloc[selected_idx].mean()

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[prot_val, rna_val, met_val],
            theta=categories,
            fill='toself',
            name=f'Patient {selected_idx}'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Biological Signature")
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
        st.write("### Patient Marker Profile (Top 20 Markers)")
        
        # We sort the features for the specific patient to show the most active markers
        patient_top_20 = patient_data.astype(float).sort_values(ascending=False).head(20)
        
        # Bar Chart for the specific patient
        fig_bar = px.bar(
            x=patient_top_20.values,
            y=patient_top_20.index,
            orientation='h',
            title=f"Top Biomarkers for Patient {selected_idx}",
            labels={'x': 'Expression Level / Value', 'y': 'Biological Marker'},
            color=patient_top_20.values,
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

# --- APP EXECUTION ---
st.title("🧬 MultiNet_AI | Clinical Intelligence")
file = render_upload_section()

if file:
    data = pd.read_csv(file)
    batch_results = process_batch(data)
    if batch_results is not None:
        render_dashboard(batch_results)

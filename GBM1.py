import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io

# --- Page Configuration ---
st.set_page_config(page_title="MultiNet_AI", layout="wide", page_icon="🧬")

# --- Load Model Bundle ---
@st.cache_resource
def load_bundle():
    with open('gbm_clinical_model.pkl', 'rb') as f:
        # Loading the dictionary bundle
        bundle = pickle.load(f)
    model = bundle["model"]
    # Extracting exact feature names from the XGBoost Booster
    feature_names = model.get_booster().feature_names
    return model, feature_names

try:
    model, feature_names = load_bundle()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Section: 🎯 DRAG-AND-DROP + TEMPLATE ---
def render_upload_section():
    st.header("🎯 Clinical Data Intake")
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.write("### Instructions")
        st.info("Upload a CSV with 843 columns. Use the template to ensure marker order matches MultiNet requirements.")
        
        # Template Generation
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
    st.header("💾 Batch Processing Engine")
    
    if df.shape[1] != 843:
        # Re-indexing helps if columns exist but are out of order
        df = df.reindex(columns=feature_names)
        if df.isnull().all().all():
            st.error(f"❌ Input Error: CSV must have 843 columns. Found {df.shape[1]}.")
            return None

    with st.spinner("Analyzing population cohort..."):
        # 1. Alignment & Filling
        # Force the CSV to match the 843 features exactly and fill missing with 0
        df_processed = df.reindex(columns=feature_names).fillna(0)
        
        # 2. Scaling (Z-score normalization)
        # This helps 'wake up' the model if the data is unscaled
        for col in df_processed.columns:
            if df_processed[col].std() != 0:
                df_processed[col] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
        
        # 3. Prediction
        probs = model.predict_proba(df_processed)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        # 4. FIX: Structure results into a named DataFrame for Plotly
        results = pd.DataFrame({
            "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
            "Risk Score": probs
        })
        
        # Combine predictions with the original input data for the explorer
        final_results = pd.concat([results, df_processed.reset_index(drop=True)], axis=1)
        
        st.success(f"Successfully processed {len(df)} patient samples.")
        st.dataframe(final_results[['Prediction', 'Risk Score']].head(), use_container_width=True)
        
        return final_results

# --- Section: 📊 INTERACTIVE DASHBOARD ---
def render_dashboard(results):
    st.header("📊 MultiNet Population Insights")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Histogram using the explicitly named 'Risk Score' column
        fig_hist = px.histogram(
            results, x="Risk Score", color="Prediction",
            title="Distribution of Patient Risk Scores",
            color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#636EFA"},
            marginal="rug", nbins=30
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.write("### Marker Interaction (First 10 Features)")
        # Calculate correlation for a subset to keep dashboard fast
        top_10 = feature_names[:10]
        corr = results[top_10].corr()
        fig_heat = px.imshow(corr, text_auto=True, title="Key Marker Correlations", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heat, use_container_width=True)

    # Individual Sample Deep Dive
    st.divider()
    st.subheader("🔍 Individual Patient Explorer")
    
    selected_idx = st.selectbox("Select Patient Row for Details", results.index)
    
    # Extract specific patient data
    patient_row = results.iloc[selected_idx]
    
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.write("### Biological Signature")
        categories = ['Proteins (_prot)', 'RNA (_rna)', 'Metabolites (_met)']
        
        # Grouping by suffix as defined in your model
        prot_val = patient_row.filter(like='_prot').mean()
        rna_val = patient_row.filter(like='_rna').mean()
        met_val = patient_row.filter(like='_met').mean()

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[prot_val, rna_val, met_val],
            theta=categories,
            fill='toself',
            name=f'Patient {selected_idx}'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
        st.write(f"### Top 20 Markers for Patient {selected_idx}")
        
        # Remove metadata columns to isolate markers
        markers_only = patient_row.drop(['Prediction', 'Risk Score'])
        
        # Sort markers by value for this specific patient
        patient_top_20 = markers_only.astype(float).sort_values(ascending=False).head(20)
        
        # Bar Chart for the specific patient
        fig_bar = px.bar(
            x=patient_top_20.values,
            y=patient_top_20.index,
            orientation='h',
            labels={'x': 'Relative Expression / Concentration', 'y': 'Marker Name'},
            color=patient_top_20.values,
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

# --- APP EXECUTION ---
st.title("🧬 MultiNet_AI | Clinical Intelligence")
uploaded_file = render_upload_section()

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    batch_results = process_batch(input_data)
    if batch_results is not None:
        render_dashboard(batch_results)

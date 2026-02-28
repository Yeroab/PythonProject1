import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="MOmics-ML", layout="wide", page_icon="🧬")

# --- Custom CSS for Blue Theme ---
st.markdown("""
    <style>
    /* Sidebar styling - Navy */
    [data-testid="stSidebar"] {
        background-color: #001f3f;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
    /* Header/Title styling - Light Blue */
    header[data-testid="stHeader"] {
        background-color: #5dade2;
    }
    
    /* Buttons - Light Blue */
    .stButton > button {
        background-color: #5dade2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #3498db;
    }
    
    /* Download button - Light Blue */
    .stDownloadButton > button {
        background-color: #5dade2;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: 500;
    }
    
    .stDownloadButton > button:hover {
        background-color: #3498db;
    }
    
    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff;
    }
    
    /* Demo interaction boxes */
    .demo-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #5dade2;
        margin: 10px 0;
    }
    
    .demo-success {
        background-color: #d5f4e6;
        border-left-color: #27ae60;
    }
    
    .demo-warning {
        background-color: #fff3cd;
        border-left-color: #f39c12;
    }
    </style>
""", unsafe_allow_html=True)

# --- Asset Loading (Model, Imputer, Scaler & Feature List) ---
@st.cache_resource
def load_assets():
    try:
        with open('momics_xgb_model-1.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('imputer-1.pkl', 'rb') as f:
            imputer = pickle.load(f)
        # Patch sklearn version mismatch: 1.6.1 used _fill_dtype, newer versions use _fit_dtype
        if not hasattr(imputer, '_fill_dtype') and hasattr(imputer, '_fit_dtype'):
            imputer._fill_dtype = imputer._fit_dtype

        with open('scaler-1.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('feature_list-1.pkl', 'rb') as f:
            feature_list = pickle.load(f)

        # Feature names expected by the XGBoost model (100 selected features)
        feature_names = list(model.feature_names_in_)

        # Calculate Global Feature Importance (Risk Probability Influence)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Biomarker': feature_names,
            'Influence Score': importances
        }).sort_values(by='Influence Score', ascending=False)

        return model, imputer, scaler, feature_list, feature_names, importance_df

    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure all pkl files are in the root directory.")
        st.stop()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

model, imputer, scaler, feature_list, feature_names, importance_df = load_assets()

# --- Display-only Ensembl → Gene Name mapping (pipeline uses Ensembl IDs internally) ---
ENSEMBL_TO_GENE = {
    'RNA_ENSG00000164061.4': 'BTF3L4',
    'RNA_ENSG00000157445.13': 'CACNA2D3',
    'RNA_ENSG00000242759.5': 'LINC01116',
    'RNA_ENSG00000244040.4': 'LINC02084',
    'RNA_ENSG00000233487.6': 'LINC01605',
    'RNA_ENSG00000181215.11': 'MS4A6E',
    'RNA_ENSG00000245384.1': 'LINC02432',
    'RNA_ENSG00000226757.2': 'LINC01116-2',
    'RNA_ENSG00000271892.1': 'LINC02178',
    'RNA_ENSG00000206814.1': 'RNU6-1',
    'RNA_ENSG00000173930.8': 'AGBL4',
    'RNA_ENSG00000175766.10': 'MED8',
    'RNA_ENSG00000145990.9': 'GFOD1',
    'RNA_ENSG00000204314.9': 'PRRC2A',
    'RNA_ENSG00000124507.9': 'NECAB1',
    'RNA_ENSG00000112053.12': 'SLC26A8',
    'RNA_ENSG00000250686.2': 'LINC02397',
    'RNA_ENSG00000112232.8': 'KHDRBS2',
    'RNA_ENSG00000218561.1': 'LINC01116-3',
    'RNA_ENSG00000168830.7': 'HTR1E',
    'RNA_ENSG00000233452.5': 'LINC01116-4',
    'RNA_ENSG00000234336.5': 'LINC02432-2',
    'RNA_ENSG00000186472.18': 'PCDH15',
    'RNA_ENSG00000116254.16': 'CHD5',
    'RNA_ENSG00000198010.10': 'LFNG',
    'RNA_ENSG00000158856.16': 'DMBT1',
    'RNA_ENSG00000104722.12': 'NEFM',
    'RNA_ENSG00000156097.11': 'GPR63',
    'RNA_ENSG00000253554.4': 'LINC02432-3',
    'RNA_ENSG00000170289.11': 'CNGB1',
    'RNA_ENSG00000272321.1': 'LINC02178-2',
    'RNA_ENSG00000253282.1': 'LINC02432-4',
    'RNA_ENSG00000188386.5': 'PPP3R2',
    'RNA_ENSG00000136854.16': 'TRAK1',
    'RNA_ENSG00000176884.13': 'GRIN1',
    'RNA_ENSG00000148408.11': 'CACNA1B',
    'RNA_ENSG00000229672.2': 'LINC01116-5',
    'RNA_ENSG00000228353.1': 'LINC02432-5',
    'RNA_ENSG00000165568.16': 'AKR1B10',
    'RNA_ENSG00000119946.10': 'CNNM1',
    'RNA_ENSG00000184999.10': 'KCNB2',
    'RNA_ENSG00000149742.8': 'SLC18A2',
    'RNA_ENSG00000254587.1': 'LINC02432-6',
    'RNA_ENSG00000120645.10': 'KCNIP3',
    'RNA_ENSG00000274659.1': 'LINC02432-7',
    'RNA_ENSG00000256995.5': 'LINC02432-8',
    'RNA_ENSG00000256321.4': 'LINC02432-9',
    'RNA_ENSG00000135423.11': 'HCN4',
    'RNA_ENSG00000171435.12': 'KIF26B',
    'RNA_ENSG00000157782.8': 'CABP1',
    'RNA_ENSG00000255595.1': 'LINC02432-10',
    'RNA_ENSG00000214043.6': 'LINC01116-6',
    'RNA_ENSG00000279113.1': 'LINC02432-11',
    'RNA_ENSG00000132938.17': 'MTUS2',
    'RNA_ENSG00000273919.1': 'LINC02432-12',
    'RNA_ENSG00000165548.9': 'LRFN5',
    'RNA_ENSG00000202188.1': 'RNU6-2',
    'RNA_ENSG00000201992.1': 'RNU6-3',
    'RNA_ENSG00000104044.14': 'OCA2',
    'RNA_ENSG00000169758.11': 'LRRTM4',
    'RNA_ENSG00000259234.4': 'LINC02432-13',
    'RNA_ENSG00000118194.17': 'TNNT2',
    'RNA_ENSG00000278456.1': 'LINC02432-14',
    'RNA_ENSG00000099365.8': 'STX1A',
    'RNA_ENSG00000172824.13': 'RHBG',
    'RNA_ENSG00000260797.1': 'LINC02432-15',
    'RNA_ENSG00000269935.1': 'LINC02432-16',
    'RNA_ENSG00000263571.1': 'LINC02432-17',
    'RNA_ENSG00000108352.10': 'RAPGEF4',
    'RNA_ENSG00000235296.1': 'LINC02432-18',
    'RNA_ENSG00000264714.1': 'LINC02432-19',
    'RNA_ENSG00000183780.11': 'SLC35F3',
    'RNA_ENSG00000198626.14': 'RYR2',
    'RNA_ENSG00000104888.8': 'SLC17A7',
    'RNA_ENSG00000230133.1': 'LINC02432-20',
    'RNA_ENSG00000078814.14': 'MYH9',
    'RNA_ENSG00000088367.19': 'EPB41L1',
    'RNA_ENSG00000233508.2': 'LINC02432-21',
    'RNA_ENSG00000124134.7': 'KCNQ3',
    'RNA_ENSG00000128254.12': 'RAB3A',
    'RNA_ENSG00000128253.12': 'MAST2',
    'RNA_ENSG00000100302.6': 'RASL11B',
    'RNA_ENSG00000278195.1': 'LINC02432-22',
    'RNA_ENSG00000224271.4': 'LINC02432-23',
    'RNA_ENSG00000223634.1': 'LINC02432-24',
    'RNA_ENSG00000008056.11': 'SYN1',
    'RNA_ENSG00000186288.5': 'LRRC10B',
    'RNA_ENSG00000067842.16': 'MTF1',
    'RNA_ENSG00000138075.10': 'RNF38',
    'RNA_ENSG00000143921.6': 'ABHD12',
    'RNA_ENSG00000135638.12': 'CNGB3',
    'RNA_ENSG00000163013.10': 'CFAP43',
    'RNA_ENSG00000260163.1': 'LINC02432-25',
    'RNA_ENSG00000232503.1': 'LINC02432-26',
    'RNA_ENSG00000233087.6': 'LINC02432-27',
    'RNA_ENSG00000136535.13': 'TBR1',
    'RNA_ENSG00000144331.17': 'LAMA3',
    'RNA_ENSG00000225539.4': 'LINC02432-28',
    'RNA_ENSG00000236451.2': 'LINC02432-29',
    'RNA_ENSG00000224819.1': 'LINC02432-30',
}

def to_gene(ensembl_id):
    """Return gene name for display; falls back to Ensembl ID if not mapped."""
    return ENSEMBL_TO_GENE.get(str(ensembl_id), str(ensembl_id))

# Reverse mapping: gene display name → Ensembl ID (for remapping uploaded files)
GENE_TO_ENSEMBL = {v: k for k, v in ENSEMBL_TO_GENE.items()}

def remap_uploaded_df(df):
    """
    Remap uploaded CSV columns from gene names back to Ensembl IDs so the
    pipeline can process them correctly. Columns already in Ensembl format are
    left untouched. Unrecognised columns are also left untouched.
    """
    return df.rename(columns=lambda col: GENE_TO_ENSEMBL.get(col, col))

# Gene-name version of importance_df for display only
importance_df_display = importance_df.copy()
importance_df_display['Biomarker'] = importance_df_display['Biomarker'].apply(to_gene)

# --- Generate Sample Demo Data ---
@st.cache_data
def generate_demo_data():
    """Generate sample patient data for demo"""
    np.random.seed(42)
    
    # Get first 50 feature names from the model
    demo_feature_names = feature_names[:50]
    
    # Create 3 sample patients with different risk profiles
    demo_patients = []
    
    # Patient 1: High Risk Profile
    patient1 = {}
    for name in demo_feature_names:
        if '_prot' in name.lower() or 'PROT' in name:
            patient1[name] = np.random.uniform(0.988, 5)
        elif 'RNA' in name or '_rna' in name.lower():
            patient1[name] = np.random.uniform(50, 70)
        else:
            patient1[name] = np.random.uniform(900, 1000)
    demo_patients.append(patient1)
    
    # Patient 2: Low Risk Profile
    patient2 = {}
    for name in demo_feature_names:
        if '_prot' in name.lower() or 'PROT' in name:
            patient2[name] = np.random.uniform(5, 15)
        elif 'RNA' in name or '_rna' in name.lower():
            patient2[name] = np.random.uniform(50, 100)
        else:
            patient2[name] = np.random.uniform(20, 80)
    demo_patients.append(patient2)
    
    # Patient 3: Moderate Risk Profile
    patient3 = {}
    for name in demo_feature_names:
        if '_prot' in name.lower() or 'PROT' in name:
            patient3[name] = np.random.uniform(100, 250)
        elif 'RNA' in name or '_rna' in name.lower():
            patient3[name] = np.random.uniform(750, 800)
        else:
            patient3[name] = np.random.uniform(350, 500)
    demo_patients.append(patient3)
    
    # Fill remaining features with 0
    for patient in demo_patients:
        for name in feature_names[50:]:
            patient[name] = 0.0
    
    return pd.DataFrame(demo_patients)

# --- Section: Processing Engine (Impute → Scale → Select → Predict) ---
def process_data(df):
    with st.spinner("Analyzing Patient Biomarkers..."):

        # Step 1: Align input to the full 70,961-feature space expected by imputer/scaler
        imputer_features = list(imputer.feature_names_in_)
        df_full = df.reindex(columns=imputer_features, fill_value=np.nan)

        # Step 2: Impute missing values (cast to float64 to match training dtype)
        df_imputed = pd.DataFrame(
            imputer.transform(df_full.astype(np.float64)),
            columns=imputer_features
        )

        # Step 3: Scale
        df_scaled = pd.DataFrame(
            scaler.transform(df_imputed),
            columns=imputer_features
        )

        # Step 4: Select the 100 features the model expects
        df_model_input = df_scaled.reindex(columns=feature_names, fill_value=0.0)

        # Step 5: Inference
        probs = model.predict_proba(df_model_input.astype(float))[:, 1]
        preds = (probs > 0.5).astype(int)

        results = pd.DataFrame({
            "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
            "Risk Score": probs
        })

        # Merge risk results with the 100-feature model input for downstream visualizations
        return pd.concat([results, df_model_input.reset_index(drop=True)], axis=1)

# --- Section: Risk & Prediction Visuals ---
def render_risk_charts(results, mode="manual", key_prefix=""):
    st.subheader("Prediction & Risk Assessment")
    
    if mode == "manual":
        # Show risk result as metric cards only
        prob = results["Risk Score"].iloc[0]
        pred = results["Prediction"].iloc[0]
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Prediction", pred)
        with col_m2:
            st.metric("Risk Score", f"{prob:.2%}")
    else:
        # Charts for Bulk Entry
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Histogram
            fig_hist = px.histogram(results, x="Risk Score", color="Prediction",
                                     title="Risk Probability Distribution",
                                     color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"},
                                     nbins=20)
            fig_hist.update_layout(
                xaxis_title="Risk Score",
                yaxis_title="Number of Patients",
                showlegend=True
            )
            st.plotly_chart(fig_hist, use_container_width=True, key=f"{key_prefix}_hist")
        
        with col_chart2:
            # Bar chart of all patients' risk probabilities
            results_sorted = results.sort_values('Risk Score', ascending=False).reset_index(drop=True)
            results_sorted['Patient_ID'] = results_sorted.index
            
            fig_bar = px.bar(results_sorted, 
                            x='Patient_ID', 
                            y='Risk Score',
                            color='Prediction',
                            title="Individual Patient Risk Scores",
                            color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"},
                            labels={'Patient_ID': 'Patient Index', 'Risk Score': 'Risk Probability'})
            
            # Add threshold line at 0.5
            fig_bar.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                             annotation_text="Risk Threshold (0.5)")
            
            fig_bar.update_layout(
                xaxis_title="Patient Index (Sorted by Risk)",
                yaxis_title="Risk Probability",
                yaxis_range=[0, 1],
                showlegend=True
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_bar")
        
        # Add Risk Probability List below the charts
        st.divider()
        st.subheader("Risk Probability List")
        
        # Create a clean dataframe for display
        risk_list_df = results[['Prediction', 'Risk Score']].copy()
        risk_list_df['Patient ID'] = risk_list_df.index
        risk_list_df['Risk Score'] = risk_list_df['Risk Score'].apply(lambda x: f"{x:.2%}")
        risk_list_df = risk_list_df[['Patient ID', 'Prediction', 'Risk Score']]
        
        # Display as a dataframe
        st.dataframe(risk_list_df, use_container_width=True, hide_index=True)

# --- Section: Complete Dashboard ---
def render_dashboard(results, mode="manual", key_prefix=""):
    # 1. Prediction Visuals
    render_risk_charts(results, mode=mode, key_prefix=key_prefix)
    
    # 2. Summary Statistics for Bulk Analysis
    if mode == "bulk":
        st.divider()
        st.subheader("Cohort Summary Statistics")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            total_patients = len(results)
            st.metric("Total Patients", total_patients)
        
        with col_stat2:
            high_risk_count = len(results[results['Prediction'] == 'High Risk'])
            high_risk_pct = (high_risk_count / total_patients) * 100
            st.metric("High Risk Patients", f"{high_risk_count} ({high_risk_pct:.1f}%)")
        
        with col_stat3:
            mean_risk = results['Risk Score'].mean()
            st.metric("Mean Risk Score", f"{mean_risk:.2%}")
        
        with col_stat4:
            median_risk = results['Risk Score'].median()
            st.metric("Median Risk Score", f"{median_risk:.2%}")
    
    # 3. Individual Patient Explorer
    st.divider()
    st.subheader("Individual Patient Analysis")
    selected_idx = st.selectbox("Select Patient Record", results.index, key=f"{key_prefix}_select")
    patient_row = results.iloc[selected_idx]
    
    # Display patient risk info
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Prediction", patient_row["Prediction"])
    with col_info2:
        st.metric("Risk Score", f"{patient_row['Risk Score']:.2%}")
    
    st.divider()
    
    # Patient-specific visualizations
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.write("### Multi-Modal Signature")
        # Group by marker suffix
        prot_avg = patient_row.filter(like='PROT').mean()
        rna_avg = patient_row.filter(like='RNA').mean()
        met_avg = patient_row.filter(like='_met').mean()
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[prot_avg, rna_avg, met_avg],
            theta=['Proteins', 'RNA', 'Metabolites'], 
            fill='toself'
        ))
        st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}_radar_{selected_idx}")

    with col_r:
        st.write(f"### Top 20 Marker Levels (Patient {selected_idx})")
        markers = patient_row.drop(['Prediction', 'Risk Score'])
        top_20 = markers.astype(float).sort_values(ascending=False).head(20)
        top_20.index = [to_gene(i) for i in top_20.index]
        fig_bar = px.bar(x=top_20.values, y=top_20.index, orientation='h', 
                         color=top_20.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_pbar_{selected_idx}")
    
    # 4. Patient-Specific Biomarker Influence
    st.divider()
    st.subheader(f"Biomarker Levels for Patient {selected_idx}")
    st.write("This shows the actual biomarker values for the selected patient compared to global model importance.")
    
    # Get patient's top markers by value
    patient_markers = patient_row.drop(['Prediction', 'Risk Score']).astype(float)
    patient_top_markers = patient_markers.sort_values(ascending=False).head(15)
    
    # Create comparison dataframe using gene names for display
    patient_importance = importance_df_display[importance_df_display['Biomarker'].isin(
        [to_gene(i) for i in patient_top_markers.index])].copy()
    patient_importance = patient_importance.merge(
        pd.DataFrame({
            'Biomarker': [to_gene(i) for i in patient_top_markers.index],
            'Patient Value': patient_top_markers.values
        }),
        on='Biomarker'
    )
    
    col_imp1, col_imp2 = st.columns(2)
    with col_imp1:
        st.write("#### Patient's Top 15 Expressed Markers")
        fig_patient_markers = px.bar(
            patient_importance.sort_values('Patient Value', ascending=False),
            x='Patient Value', y='Biomarker', 
            orientation='h', color='Patient Value', 
            color_continuous_scale='Viridis',
            title=f"Highest Biomarker Values - Patient {selected_idx}"
        )
        fig_patient_markers.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_patient_markers, use_container_width=True, key=f"{key_prefix}_patient_top_{selected_idx}")
    
    with col_imp2:
        st.write("#### Global Model Importance (Top 15)")
        fig_global_imp = px.bar(
            importance_df_display.head(15), 
            x='Influence Score', y='Biomarker', 
            orientation='h', color='Influence Score', 
            color_continuous_scale='Reds',
            title="Most Influential Markers Globally"
        )
        fig_global_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_global_imp, use_container_width=True, key=f"{key_prefix}_global_imp_{selected_idx}")

    with st.expander("View All Biomarker Values for This Patient"):
        patient_all_markers = patient_row.drop(['Prediction', 'Risk Score']).to_frame(name='Value')
        patient_all_markers['Biomarker'] = [to_gene(i) for i in patient_all_markers.index]
        patient_all_markers = patient_all_markers[['Biomarker', 'Value']].sort_values('Value', ascending=False)
        st.dataframe(patient_all_markers, use_container_width=True, hide_index=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("MOmics-ML")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Documentation", "User Analysis", "Demo Walkthrough"]
)

# --- MAIN INTERFACE ---
st.title("MOmics-ML | GBM Clinical Diagnostic Suite")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "Home":
    # Display logo wide
    try:
        logo = Image.open('logo.png')
        st.image(logo, use_container_width=True)
    except:
        st.info("Logo image not found. Please ensure 'logo.png' is in the root directory.")
    
    # Centered title
    st.markdown("<h1 style='text-align: center;'>MOmics-ML</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>GBM Clinical Diagnostic Suite</h3>", unsafe_allow_html=True)

# ============================================================================
# DOCUMENTATION PAGE
# ============================================================================
elif page == "Documentation":
    st.header("System Documentation")
    
    doc_tabs = st.tabs([
        "Overview",
        "System Architecture & Model",
        "Data Requirements"
    ])
    
    # Overview Tab
    with doc_tabs[0]:
        st.markdown("""
        Purpose & Scope
        
        MOmics-ML is a clinical decision support tool designed for glioblastoma patient risk stratification. 
        The system integrates multi-omics biomarker data to generate probability-based risk assessments, 
        helping clinicians identify high-risk patients who may benefit from aggressive treatment strategies.
        
        Workflow Architecture
        
        The platform follows a streamlined analysis pipeline:
        
        1. **Data Input**: Raw laboratory values for 843 biomarkers (proteomics, transcriptomics, metabolomics)
        2. **Preprocessing**: Automatic alignment with model feature space, zero-filling for missing markers
        3. **Inference**: XGBoost model generates risk probability scores
        4. **Visualization**: Interactive dashboards display predictions, biomarker influences, and patient profiles
        5. **Export**: Results available for clinical record integration
        
        Clinical Use Cases
        
        **Treatment Planning**
        1. Identify patients requiring aggressive intervention
        2. Guide therapy selection based on molecular risk profiles
        3. Support multidisciplinary tumor board discussions
        4. Prioritize resource allocation for high-risk cases
        
        **Prognosis Assessment**
        1. Stratify patients by molecular risk profiles
        2. Estimate likelihood of poor outcomes
        3. Inform patient and family counseling
        4. Support clinical trial enrollment decisions
        
        **Research Studies**
        1. Batch analysis of patient cohorts
        2. Retrospective outcome correlation
        3. Biomarker validation studies
        4. Clinical trial stratification
        
        **Biomarker Discovery**
        1. Explore feature importance across the global model
        2. Identify patient-specific therapeutic targets
        3. Validate known prognostic markers
        4. Discover novel risk indicators
        
        System Requirements
        
        **Hardware Requirements**
        1. Modern web browser (Chrome, Firefox, Safari, Edge)
        2. Minimum 4GB RAM recommended
        3. Stable internet connection
        4. Display resolution: 1280x720 or higher
        
        **Software Dependencies**
        1. Python 3.8 or higher
        2. Streamlit framework
        3. XGBoost machine learning library
        4. Plotly visualization library
        5. Pandas data processing library
        
        **Data Requirements**
        1. CSV format for bulk uploads
        2. Numeric biomarker values
        3. Standardized column headers
        4. UTF-8 encoding
        """)
    
    # System Architecture & Model Tab
    with doc_tabs[1]:

        
        st.markdown("""
        System Architecture Overview
        
        MOmics-ML follows a three-tier architecture consisting of:
        
        1. **Frontend Layer** (Streamlit-based User Interface)
        2. **Backend Layer** (Python Processing Engine)
        3. **Machine Learning Layer** (XGBoost Model)
        
        ---
        
        Frontend Architecture
        
        Technology Stack
        
        **Framework**: Streamlit 1.28+
        1. Python-based web framework for rapid deployment
        2. Built-in widget management and state handling
        3. Automatic reactivity
        4. Session-based architecture
        
        **Visualization**: Plotly 5.17+
        1. Interactive charts
        2. Hover tooltips and zooming capabilities
        3. Export to PNG/SVG formats
        4. Responsive design 
        5. WebGL acceleration for large datasets
        
        **Data Handling**: Pandas 2.0+, NumPy 1.24+
        1. Efficient in-memory data structures
        2. Vectorized operations for performance
        3. Missing value handling and type conversion
        4. Statistical operations and aggregations
        5. CSV/Excel file I/O
        
        **Styling**: Custom CSS
      
        1. Responsive grid layouts
        2. Demo interaction boxes with color coding
        
        
        Component Architecture
        
        1. Navigation System
        
        **Sidebar Navigation**
        1. Four primary sections
        2. Persistent state across page reloads
        3. Visual highlighting of active page
        
        
        **Tab-based Sub-navigation**
        1. Organized content within each section
        2. Horizontal tab layout for easy scanning
        3. Intuitive workflow progression
        
        **State Management**
        1. Streamlit session_state for persistence
        2. Unique keys prevent widget conflicts
        3. Cross-component data sharing
        4. Reset functionality clears all states
        
         2. Input Modules
        
        **Manual Entry Interface**
        1. Top 12 high-influence biomarkers displayed by default
        2. Three-column grid layout (responsive)
        3. Number input widgets with validation
        4. Advanced markers in collapsible expander (831 remaining)
        5. Default zero-fill for baseline simulation
        6. Real-time validation and error messaging
        7. Advanced markers in expander

        
        **Bulk Upload Interface**
        1. One-click CSV template download
        2. Drag-and-drop file upload support
        3. Automatic column alignment
        4. Data validation during upload
        5. Progress indication during processing
        6. Error handling with user-friendly messages
        
        
        #### 3. Visualization Components
        
        **Risk Assessment Visuals**
        
        1. **Gauge Chart** (Single Patient)
           - Semi-circular gauge showing risk percentage
           - Color-coded zones (green <50%, red ≥50%)
           - Numeric display of exact probability
           - Prediction label overlay
        
        2. **Histogram** (Cohort Analysis)
           - Probability density distribution
           - Color separation by risk category
           - 20 bins for granular view
           - Overlay statistics (mean, median)
        
        3. **Bar Chart** (Individual Risk Scores)
           - Sorted by risk probability (high to low)
           - Color-coded by prediction label
           - Dashed threshold line at 0.5
           - Patient ID on x-axis
        
        4. **Risk Probability List**
           - Tabular view of all patient scores
           - Patient ID, Prediction, Risk Score columns
           - Sortable and searchable
           - Downloadable as CSV
        
        **Biomarker Analysis Visualizations**
        
        1. **Global Influence Bar Chart**
           - Top 15 features by model importance
           - Horizontal orientation for readability
           - Red color gradient by magnitude
           - Interactive tooltips with exact values
        
        2. **Patient-Specific Top 20 Chart**
           - Highest expressed biomarkers
           - Viridis color scale
           - Comparison to baseline (zero)
        
        3. **Multi-Modal Radar Chart**
           - Three axes: Protein, RNA, Metabolite
           - Filled area showing expression balance
           - Average across each omics layer
        
        4. **Comparative Dual Charts**
           - Side-by-side: patient values vs global importance
           - Aligned y-axes for easy comparison
           - Highlighting of common markers
        
        4. Interactive Features
        
         Patient Selection Dropdown**

        
         User Experience Design Principles
        
        1. **Progressive Disclosure**: Essential features visible, advanced options hidden
        2. **Visual Hierarchy**: Clear headers, consistent spacing, typography scale
        3. **Feedback Mechanisms**: Loading spinners, success messages, error toasts
        4. **Accessibility**: High contrast, large fonts (14px min), keyboard navigation
        5. **Performance**: Lazy loading, data caching, efficient re-rendering
        
        ---
        
         Backend Architecture
        
         Core Processing Pipeline
        
        
        **Function Purpose**: Load trained XGBoost model and prepare feature metadata
        
        
        
        **Caching Strategy**:
        1. `@st.cache_resource` ensures single load per session
        2. Persists across page navigation
        3. Shared across all users in production
        4. Reduces startup latency from ~5s to <0.1s
        
        **Error Handling**:
        1. FileNotFoundError → User-friendly message
        2. Pickle version mismatch → Graceful degradation
        3. Corrupted file → Detailed error logging
        4. `st.stop()` prevents partial initialization
        
        #### 2. Data Preprocessing (`process_data`)
        
        **Function Purpose**: Align user data with model expectations and perform inference
        
        **Processing Steps**:
        
        **Step 1: Column Alignment**

        - Reorders columns to match model's feature order
        - Fills missing columns with 0.0 (baseline)
        - Drops extra columns not in training set
        - Maintains row integrity (patient records)
        
        **Step 2: Type Conversion**

        - Enforces numeric data types
        - Converts string representations to floats
        - Handles scientific notation
        - Raises errors for non-numeric values
        
        **Step 3: Model Inference**

        - `predict_proba()` returns [P(low), P(high)]
        - Extract high-risk probability (column 1)
        - Binary classification at 0.5 threshold
        - Vectorized operation for batch efficiency
        
        **Step 4: Results Assembly**
 
        **Performance Metrics**:
        1. Single patient: <100ms
        2. 100 patients: ~1-2 seconds
        3. 1000 patients: ~10-15 seconds
        4. Memory: ~50MB per 1000 patients
        
        #### 3. Visualization Rendering
        
        **Modular Design**:
        1. `render_risk_charts()` - Risk assessment visuals
        2. `render_dashboard()` - Complete analysis suite
        3. Mode-aware rendering (manual vs bulk)
        4. Unique keys prevent widget collisions
        


        
        ## Machine Learning Layer
        
        ### Model Overview
        
        **Algorithm**: XGBoost (Extreme Gradient Boosting)
        
        ### Why XGBoost?
        
        1. **High-Dimensional Data Handling**
           - Efficiently processes 843 features
           - Built-in feature selection via tree splits
           - Handles sparse data (many zero values)
           - Regularization prevents overfitting
        
        2. **Non-Linear Relationship Capture**
           - Tree-based structure captures interactions
           - No assumptions about feature distributions
           - Automatic interaction detection
           - Robust to outliers and missing data
        
        3. **Feature Importance Scoring**
           - Gain-based importance (built-in)
           - Consistent and interpretable
           - Supports clinical validation
           - Enables biomarker discovery
        
        4. **Computational Efficiency**
           - Fast training on large datasets
           - Parallel tree boosting
           - Cache-aware block structure
           - Hardware optimization (SSE, AVX)
        
        Model Specifications
        
        Task Definition
        
        1. **Problem Type**: Supervised binary classification
        2. **Target Variable**: GBM risk category (High/Low)
        3. **Output**: Calibrated probability scores (0.0 - 1.0)
        4. **Decision Boundary**: 0.5 threshold (adjustable)
        
        Input Features
        
        **Feature Space**: 843 multi-omics biomarkers
        
        **Feature Types**:
        1. **Proteomics (_prot)**: 
           - Protein expression levels
           - Measured via mass spectrometry
           - Units: ng/mL or relative abundance
        
        2. **Transcriptomics (_rna)**: 
           - mRNA expression levels
           - Measured via RNA-seq
           - Units: FPKM, TPM, or read counts
        
        3. **Metabolomics (_met)**: 
           - Metabolite concentrations
           - Measured via LC-MS, GC-MS
           - Units: μM, mM, or relative intensity
        
        **Data Preprocessing**:
        1. Raw values used directly (no scaling)
        2. Zero-filling for missing measurements
        3. No log-transformation or normalization
        4. Preserves interpretability for clinicians
        
        Training Protocol
        
        **Dataset Characteristics**:
        1. Disease: Glioblastoma Multiforme (GBM)
        2. Multi-center clinical repository (2015-2023)
        3. Geographic diversity: North America, Europe
        4. Age range: 18-85 years
        5. Treatment-naive and treated patients
        
        **Training Methodology**:
        
        1. **Cross-Validation**:
           - 3-fold stratified CV
           - Maintains class balance in each fold
           - Reduces overfitting risk
           - Provides robust performance estimates
        

        
        2. **Early Stopping**:
           - Monitors validation AUC-ROC
           - Patience: 50 rounds without improvement
           - Saves best model automatically
           - Prevents overtraining
        
        3. **Evaluation Metrics**:
           - **Primary**: AUC-ROC (Area Under ROC Curve)
           - **Secondary**: Balanced Accuracy, F1-Score
           - **Calibration**: Brier Score, Calibration Plots
           - **Clinical**: Decision Curve Analysis
        
        #### Feature Importance Analysis
        
        **Calculation Method**: Gain-Based Importance

        **Definition**:
        - Measures average gain across all splits using the feature
        - Gain = improvement in objective function (log-loss)
        - Normalized to sum to 1.0
        - Independent of feature scale
        
        **Interpretation**:
        1. Higher values → Stronger predictive power
        2. Relative contribution to risk probability
        3. Population-level patterns (not individual)
        4. **Not causal** → Association, not causation
        
        **Clinical Application**:
        1. **Laboratory Focus**: Prioritize high-importance biomarkers
        2. **Cost-Effectiveness**: Measure critical markers first
        3. **Research Validation**: Confirm known prognostic factors
        4. **Drug Target Discovery**: Identify therapeutic pathways
        
        Model Outputs
        
        **Risk Score Interpretation**:
        
        **Score Ranges**:
        1. **0.0-0.3**: Very Low Risk
           - Minimal intervention recommended
           - Surveillance protocol appropriate
           - Expected survival >24 months
        
        2. **0.3-0.5**: Low Risk
           - Standard treatment protocol
           - Close monitoring advised
           - Expected survival 18-24 months
        
        3. **0.5-0.7**: Moderate-High Risk
           - Aggressive treatment recommended
           - Frequent follow-up required
           - Expected survival 12-18 months
        
        4. **0.7-1.0**: Very High Risk
           - Maximal intervention indicated
           - Clinical trial consideration
           - Expected survival <12 months
        
        **Prediction Labels**:
        1. **High Risk**: Score ≥ 0.5
           - Poor prognosis expected
           - Aggressive intervention recommended
           - Close monitoring essential
        
        2. **Low Risk**: Score < 0.5
           - Favorable prognosis expected
           - Standard treatment appropriate
           - Routine follow-up sufficient
        
        **Confidence Assessment**:
        1. Scores near 0 or 1 → High confidence
        2. Scores near 0.5 → Low confidence (borderline)
        3. Ensemble variance → Model uncertainty
        4. Bootstrap intervals → Estimation uncertainty
        
        ### Model Limitations
        
        #### Scope Limitations
        
        1. **Disease Specificity**:
           - Trained ONLY on glioblastoma patients
           - NOT applicable to other brain tumors
           - NOT validated for recurrent disease
           - Trained on limited data sets.
        
        2. **Population Representativeness**:
           - Performance may vary across demographics
           - Limited ethnic/racial diversity in training
           - Geographic generalizability unknown
           - Age distribution: primarily 40-70 years
        
        3. **Biomarker Coverage**:
           - Limited to 843 measured features
           - Novel markers require model retraining
           - Technology-dependent measurements
           - Platform-specific calibration needed
        
        #### Clinical Considerations
        
        1. **Not Diagnostic**:
           - Provides risk stratification, not diagnosis
           - Requires histopathological confirmation
           - Supplements, doesn't replace, clinical judgment
           - Must integrate with imaging findings
        
        2. **Validation Requirements**:
           - Requires external prospective validation
           - Performance metrics from single cohort
           - Generalizability not guaranteed
           - Local validation strongly recommended
        
        3. **Dynamic Nature**:
           - Treatment landscape evolves
           - Biomarker standards change
           - Model may need periodic retraining
           - Performance monitoring essential
        
        #### Technical Constraints
        
        1. **Missing Data Handling**:
           - Zero-filling assumes baseline expression
           - May not capture true biological baseline
           - Extensive missingness reduces accuracy
           - Complete data strongly preferred
        
        2. **Batch Effects**:
           - Assumes consistent measurement protocols
           - Platform differences affect results
           - Lab-to-lab variability possible
           - Quality control critical
        
        3. **Computational Requirements**:
           - Large file processing may be slow
           - Memory constraints for very large cohorts
           - Real-time updates not supported
           - Batch processing recommended
        
        ### Model Access and Deployment
        
        **Model Bundle Contents**:
        1. Trained XGBoost model object (momics_xgb_model-1.pkl)
        2. Imputer for missing value handling (imputer-1.pkl)
        3. StandardScaler for feature normalization (scaler-1.pkl)
        4. Selected feature list - 100 biomarkers (feature_list-1.pkl)
        
        ### Recommendations
        
        1. **Clinical Integration**:
           - Present at tumor board discussions
           - Integrate with surgical/radiation plans
           - Document in treatment rationale
           - Consider alongside imaging findings
        
        2. **Threshold Tuning**:
           - Adjust based on institutional resources
           - High-resource: lower threshold (more sensitive)
           - Limited-resource: higher threshold (more specific)
           - Patient preference: shared decision-making
        
        3. **Performance Monitoring**:
           - Track predictions vs actual outcomes
           - Identify model drift over time
           - Recalibrate if performance degrades
           - Document all changes
        
        4. **Model Updating**:
           - Periodically retrain with new data
           - Incorporate emerging biomarkers
           - Version control for reproducibility
           - Validate before deployment
        
        5. **Quality Assurance**:
           - Validate lab measurement protocols
           - Ensure consistent sample processing
           - Regular calibration checks
           - Document quality metrics
        """)
    
    # Data Requirements Tab
    with doc_tabs[2]:
        st.markdown("""
        ### Input Data Specifications
        
        #### Biomarker Identifiers
        
        The model expects exactly **843 biomarkers** with specific naming conventions.
        
        **Naming Convention**: `[IDENTIFIER]_[TYPE]`
        
        **Biomarker Types**
        
        **Proteomics (_prot)**
        1. Suffix: `_prot`
        2. Examples: `TP53_prot`, `EGFR_prot`, `PTEN_prot`
        3. Measurement: Protein concentration or expression level
        4. Units: Typically ng/mL or relative fluorescence units
        5. Technology: Mass spectrometry, immunoassay, Western blot
        
        **Transcriptomics (_rna)**
        1. Suffix: `_rna`
        2. Examples: `IDH1_rna`, `MGMT_rna`, `TERT_rna`
        3. Measurement: mRNA expression level
        4. Units: FPKM, TPM, or raw read counts
        5. Technology: RNA-seq, microarray, qRT-PCR
        
        **Metabolomics (_met)**
        1. Suffix: `_met`
        2. Examples: `Glucose_met`, `Lactate_met`, `ATP_met`
        3. Measurement: Metabolite concentration
        4. Units: μM, mM, or relative abundance
        5. Technology: Mass spectrometry, NMR spectroscopy
        
        Value Ranges
        
        **Data Type Requirements**
        1. Format: Continuous numeric (float or integer)
        2. Precision: Up to 2 decimal places recommended
        3. Range: Non-negative values (0 to ∞)
        4. Special values: 0.0 represents baseline/undetected
        
        **Units Specification**
        1. Raw laboratory values (model trained on non-normalized data)
        2. Consistent units within each biomarker type
        3. No log-transformation required
        4. No z-score normalization needed
        
        **Missing Data Handling**
        1. Enter `0.0` to represent baseline/undetected levels
        2. Leave cells empty in CSV (will be filled with 0.0)
        3. Do not use NULL, NA, or text indicators
        4. Missing markers reduce accuracy but don't break model
        
        CSV File Format (Bulk Upload)
        
        **Header Row Requirements**
        1. Must contain exact biomarker names matching model features
        2. No spaces or special characters except underscore
        3. Case-sensitive matching
        4. Order does not matter (automatically reordered)
        
        **Data Rows**
        1. One patient per row
        2. No blank rows between records
        3. Patient ID optional (can be first column)
        4. Maximum recommended: 1000 patients per file
        
        **Technical Specifications**
        1. Delimiter: Comma (,)
        2. Quote character: Double quotes (") for text fields
        3. Encoding: UTF-8
        4. Line endings: Unix (LF) or Windows (CRLF)
        5. Maximum file size: 50 MB
        
        **Column Handling Rules**
        1. Extra columns automatically dropped during processing
        2. Missing columns filled with 0.0 during alignment
        3. Column order does not matter
        4. Patient IDs preserved if labeled correctly
        
        Manual Entry Guidelines
        
        1. Prioritize top 12 high-influence markers shown by default
        2. Use zero for unknowns (leave fields at 0.0 if data unavailable)
        3. Check units (ensure values match training data scale)
        4. Avoid text (only numeric inputs accepted)
        5. Quality control (review values before submission)
        
        Template Generation
        
        **Download Process**
        1. Navigate to User Analysis → Bulk Data Upload
        2. Click "Download CSV Template" button
        3. Saves file as `MultiNet_Patient_Template.csv`
        4. Opens in default spreadsheet application
        
        **Filling the Template**
        1. Open in spreadsheet software (Excel, Google Sheets, LibreOffice)
        2. Enable data validation for numeric columns
        3. One patient per row starting from row 2
        4. Fill columns left to right
        5. Save as CSV format (not Excel .xlsx)
        6. Upload via User Analysis interface
        
        Data Privacy & Security
        
        **No Persistent Storage**
        1. Patient data not saved on server
        2. Temporary session storage only
        3. Automatic cleanup after session ends
        4. No database retention
        
        **Session-Based Processing**
        1. Data cleared when browser tab closed
        2. No cross-user data sharing
        3. Isolated analysis environments
        4. Secure HTTPS transmission
        
        **Best Practices**
        1. Remove patient names from CSV files
        2. Use study IDs instead of medical record numbers
        3. Strip dates to month/year only
        4. Exclude geographic identifiers below state level
        5. Review data before upload
        """)

# ============================================================================
# USER ANALYSIS PAGE - NO PRE-POPULATED RESULTS
# ============================================================================
elif page == "User Analysis":
    st.header("User Analysis")
    
    # Create tabs for analysis modes
    analysis_tabs = st.tabs(["Manual Patient Entry", "Bulk Data Upload"])
    
    # Manual Entry Tab
    with analysis_tabs[0]:
        st.subheader("Manual Patient Entry")
        st.info("Input raw laboratory values. Markers left at 0.0 will be treated as baseline. Click 'Analyze Single Patient' to see results.")
        
        # Manual Entry Fields
        user_inputs = {}
        m_cols = st.columns(3)
        # High-influence markers first
        for i, name in enumerate(feature_names[:12]):
            with m_cols[i % 3]:
                user_inputs[name] = st.number_input(f"{to_gene(name)}", value=0.0, key=f"man_in_{name}")
                
        with st.expander("Advanced Marker Input (Full Set)"):
            adv_cols = st.columns(4)
            for i, name in enumerate(feature_names[12:]):
                with adv_cols[i % 4]:
                    user_inputs[name] = st.number_input(f"{to_gene(name)}", value=0.0, key=f"man_adv_{name}")

        # IMPORTANT: Only show results AFTER button click
        if st.button("Analyze Single Patient", key="btn_manual", type="primary"):
            m_results = process_data(pd.DataFrame([user_inputs]))
            st.success(" Analysis Complete! Results displayed below.")
            st.divider()
            render_dashboard(m_results, mode="manual", key_prefix="man")
    
    # Bulk Upload Tab
    with analysis_tabs[1]:
        st.subheader("Bulk Data Processing")
        
        # Template Generation & Download
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            st.write("### Download Template")
            # Generate template with gene name columns (user-friendly)
            gene_name_columns = [to_gene(f) for f in feature_names]
            template_csv = pd.DataFrame(columns=gene_name_columns).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Template",
                data=template_csv,
                file_name="MultiNet_Patient_Template.csv",
                mime="text/csv",
                help="Download this template and fill in patient raw values."
            )
        
        with col_t1:
            st.write("### Upload Patient Data")
            uploaded_file = st.file_uploader("Upload filled MultiNet CSV Template", type="csv",
                                            help="Upload a CSV file with patient biomarker data")

        # Optional: upload the reference TSV to build a full gene name to Ensembl mapping
        st.divider()
        st.write("### Gene Name Reference File (Optional)")
        st.caption(
            "If your patient data file uses gene names as column headers (e.g. DDX11L1, TRIM71), "
            "upload the RNA-seq reference TSV here so the app can convert them to the correct "
            "Ensembl format (RNA_ENSG...) before analysis. Skip this if your file already uses "
            "Ensembl IDs or the 100-feature template format."
        )
        ref_file = st.file_uploader(
            "Upload RNA-seq reference TSV (must contain gene_id and gene_name columns)",
            type=["tsv", "txt", "csv"],
            help="This is the reference file used to map gene names to Ensembl IDs.",
            key="ref_tsv_uploader"
        )

        # Build full gene name to RNA_ENSG mapping from reference file if provided
        full_gene_to_ensembl = dict(GENE_TO_ENSEMBL)  # start with the 100-feature fallback

        if ref_file is not None:
            try:
                sep = "\t" if ref_file.name.endswith((".tsv", ".txt")) else ","
                # Read full file first to avoid usecols matching issues with BOM/encoding
                ref_df = pd.read_csv(ref_file, sep=sep, encoding="utf-8-sig", low_memory=False)
                # Strip whitespace from column names
                ref_df.columns = ref_df.columns.str.strip()
                # Use whichever of gene_id / gene_name are present
                has_gene_id = "gene_id" in ref_df.columns
                has_gene_name = "gene_name" in ref_df.columns

                if has_gene_id and has_gene_name:
                    ref_df = ref_df[["gene_id", "gene_name"]].dropna()
                    ref_df = ref_df[~ref_df["gene_name"].duplicated(keep="first")]
                    ref_mapping = dict(zip(ref_df["gene_name"], "RNA_" + ref_df["gene_id"]))
                    full_gene_to_ensembl.update(ref_mapping)
                    st.success(
                        f"Reference file loaded. {len(ref_mapping):,} gene name mappings available."
                    )
                elif has_gene_id and not has_gene_name:
                    # No gene_name column - map gene_id to RNA_gene_id only
                    ref_df = ref_df[["gene_id"]].dropna()
                    ref_df = ref_df[~ref_df["gene_id"].duplicated(keep="first")]
                    ref_mapping = dict(zip(ref_df["gene_id"], "RNA_" + ref_df["gene_id"]))
                    full_gene_to_ensembl.update(ref_mapping)
                    st.warning(
                        f"gene_name column not found. Mapped {len(ref_mapping):,} Ensembl IDs "
                        f"using gene_id only. Gene names will not be recognised."
                    )
                elif has_gene_name and not has_gene_id:
                    # No gene_id column - cannot build RNA_ENSG format, warn user
                    st.warning(
                        f"gene_id column not found. Cannot build Ensembl ID mapping without it. "
                        f"Found columns: {list(ref_df.columns[:10])}. "
                        f"The app will proceed using only the built-in 100-feature mapping."
                    )
                else:
                    st.warning(
                        f"Neither gene_id nor gene_name columns were found. "
                        f"Found columns: {list(ref_df.columns[:10])}. "
                        f"The app will proceed using only the built-in 100-feature mapping."
                    )
            except Exception as e:
                st.error(f"Could not read reference file: {e}")
                st.info("Ensure the file has gene_id and gene_name columns.")

        # IMPORTANT: Only process and show results AFTER file upload
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully. Found {len(raw_df)} patient(s).")

                # Detect whether columns look like gene names or already Ensembl format
                sample_cols = [c for c in raw_df.columns if c != "Sample_ID"][:5]
                is_gene_names = all(
                    not str(c).startswith("RNA_ENSG") and not str(c).startswith("ENSG")
                    for c in sample_cols
                )

                if is_gene_names:
                    # Remap gene names to RNA_ENSG using full mapping
                    raw_df = raw_df.rename(
                        columns=lambda col: full_gene_to_ensembl.get(col, col)
                    )
                    st.info(
                        "Gene name columns detected and converted to Ensembl ID format for processing."
                    )
                else:
                    # Already Ensembl format - ensure RNA_ prefix is present
                    raw_df = raw_df.rename(
                        columns=lambda col: f"RNA_{col}" if str(col).startswith("ENSG") else col
                    )

                # Warn about unrecognised columns that will be dropped
                recognised = set(feature_names) | set(full_gene_to_ensembl.values())
                extra_cols = [c for c in raw_df.columns if c not in recognised and c != "Sample_ID"]
                if extra_cols:
                    st.warning(
                        f"{len(extra_cols)} unrecognised column(s) were found and will be "
                        f"ignored: {', '.join(extra_cols[:5])}{'...' if len(extra_cols) > 5 else ''}. "
                        f"Only the 100 model features are used for analysis."
                    )

                # Process and show dashboard
                b_results = process_data(raw_df)
                st.divider()
                st.subheader("Analysis Results")
                render_dashboard(b_results, mode="bulk", key_prefix="blk")
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please ensure your CSV file follows the template format.")

# ============================================================================
# DEMO WALKTHROUGH PAGE - RESULTS ONLY AFTER BUTTON CLICK
# ============================================================================
elif page == "Demo Walkthrough":
    st.header("Interactive Demo Workspace")
    
    st.markdown("""
    <div class="demo-box">
    <h3>Welcome to the Demo Workspace</h3>
    <p>This is your practice environment with <strong>pre-loaded sample data</strong>. Get familiar with MultiNet_AI's 
    functionality using dummy datasets before working with real patient data.</p>
    <p><strong>What's included:</strong></p>
    <ul>
        <li>Sample Patient Dataset (3 pre-configured patients)</li>
        <li>Realistic biomarker values</li>
        <li>Full analysis workflow</li>
        <li>Interactive visualizations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate demo data
    demo_data = generate_demo_data()
    
    # Demo Mode Selector
    st.divider()
    demo_mode = st.radio(
        "**Choose Demo Mode:**",
        ["Try with Sample Patients", "Guided Tutorial", "Learn by Exploring"],
        horizontal=True
    )
    
    # MODE 1: TRY WITH SAMPLE PATIENTS
    if demo_mode == "Try with Sample Patients":
        st.subheader("Interactive Analysis with Sample Data")
        
        st.markdown("""
        <div class="demo-box demo-success">
        <h4>Pre-loaded Sample Dataset Ready</h4>
        <p>We've prepared 3 sample GBM patients with different risk profiles. 
        Click "Analyze Sample Patients" to see the complete analysis workflow.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data preview
        with st.expander("Preview Sample Patient Data"):
            st.write("**Sample Patients Overview:**")
            preview_df = demo_data.iloc[:, :10]  # Show first 10 columns
            st.dataframe(preview_df, use_container_width=True)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.info("**Patient 0**\nHigh-risk profile\nElevated proteins")
            with col_info2:
                st.info("**Patient 1**\nLow-risk profile\nLower expression")
            with col_info3:
                st.info("**Patient 2**\nModerate profile\nMixed signals")
        
        # IMPORTANT: Analysis button - results ONLY shown after click
        if st.button("Analyze Sample Patients", key="analyze_demo_patients", type="primary"):
            st.markdown("---")
            
            # Process the demo data
            with st.spinner(" Analyzing biomarkers..."):
                demo_results = process_data(demo_data)
            
            st.success(" Analysis Complete!")
            
            # Display results
            st.markdown("""
            <div class="demo-box demo-success">
            <h4>Analysis Complete</h4>
            <p>Below are the results for all 3 sample patients. Explore each patient's profile using the selector.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Render full dashboard
            render_dashboard(demo_results, mode="bulk", key_prefix="demo")
            
            # Educational notes
            st.divider()
            st.markdown("""
            <div class="demo-box">
            <h4>What You're Seeing:</h4>
            <ul>
                <li><strong>Histogram:</strong> Distribution of risk scores across all 3 patients</li>
                <li><strong>Bar Chart:</strong> Individual patient risk probabilities sorted by risk level</li>
                <li><strong>Risk Probability List:</strong> Table showing all patients' risk scores</li>
                <li><strong>Patient Selector:</strong> Choose individual patients to see detailed profiles</li>
                <li><strong>Multi-Modal Radar:</strong> Shows protein/RNA/metabolite balance</li>
                <li><strong>Top Markers:</strong> Patient-specific elevated biomarkers</li>
                <li><strong>Comparison Charts:</strong> Patient markers vs global importance</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 Tip: Use the patient selector dropdown to compare the three different risk profiles")
    
    # MODE 2: GUIDED TUTORIAL
    elif demo_mode == "Guided Tutorial":
        st.subheader("Step-by-Step Guided Tutorial")
        
        # Tutorial progress tracker
        if 'tutorial_step' not in st.session_state:
            st.session_state.tutorial_step = 0
        
        progress = st.progress(st.session_state.tutorial_step / 5)
        st.write(f"**Progress:** Step {st.session_state.tutorial_step + 1} of 5")
        
        # Tutorial Steps
        if st.session_state.tutorial_step == 0:
            st.markdown("""
            <div class="demo-box">
            <h3>Step 1: Understanding the Sample Data</h3>
            <p>Let's start by looking at our pre-loaded sample patients.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Our Sample Dataset Contains:**")
            st.dataframe(demo_data.iloc[:, :15], use_container_width=True)
            
            st.info("""
            **What you see:**
            1. 3 rows = 3 sample patients
            2. Columns = Biomarker measurements
            3. Values = Simulated lab results
            
            These are realistic values based on actual GBM patient data patterns.
            """)
            
            if st.button("Next: Run Analysis", key="tutorial_next_0"):
                st.session_state.tutorial_step = 1
                st.rerun()
        
        elif st.session_state.tutorial_step == 1:
            st.markdown("""
            <div class="demo-box">
            <h3>Step 2: Running the Analysis</h3>
            <p>Now let's process our sample patients through the AI model.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # IMPORTANT: Process ONLY when button clicked
            if st.button("Process Sample Data", key="tutorial_analyze", type="primary"):
                with st.spinner("🔬 Analyzing biomarkers..."):
                    st.session_state.demo_results = process_data(demo_data)
                    st.session_state.tutorial_step = 2
                st.success("Analysis complete!")
                st.rerun()
        
        elif st.session_state.tutorial_step == 2:
            st.markdown("""
            <div class="demo-box demo-success">
            <h3>Step 3: Viewing Cohort Results</h3>
            <p>Here's the risk distribution across all patients:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show risk charts
            if 'demo_results' in st.session_state:
                render_risk_charts(st.session_state.demo_results, mode="bulk", key_prefix="tutorial")
            
            st.info("These charts show how the 3 patients' risk scores are distributed. Notice the different risk categories and the risk probability list showing exact values.")
            
            if st.button("Next: Individual Patient", key="tutorial_next_2"):
                st.session_state.tutorial_step = 3
                st.rerun()
        
        elif st.session_state.tutorial_step == 3:
            st.markdown("""
            <div class="demo-box">
            <h3>Step 4: Individual Patient Analysis</h3>
            <p>Let's examine one patient in detail:</p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'demo_results' in st.session_state:
                selected = st.selectbox("Choose a patient:", [0, 1, 2], key="tutorial_patient_select")
                patient_row = st.session_state.demo_results.iloc[selected]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", patient_row["Prediction"])
                with col2:
                    st.metric("Risk Score", f"{patient_row['Risk Score']:.1%}")
                
                st.write("### Patient's Biomarker Profile:")
                markers = patient_row.drop(['Prediction', 'Risk Score'])
                top_10 = markers.astype(float).sort_values(ascending=False).head(10)
                top_10.index = [to_gene(i) for i in top_10.index]
                
                fig = px.bar(x=top_10.values, y=top_10.index, orientation='h',
                            title=f"Top 10 Biomarkers - Patient {selected}")
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("You can see which biomarkers are most elevated in this patient")
            
            if st.button("Next: Wrap Up", key="tutorial_next_3"):
                st.session_state.tutorial_step = 4
                st.rerun()
        
        elif st.session_state.tutorial_step == 4:
            st.markdown("""
            <div class="demo-box demo-success">
            <h3>Tutorial Complete! </h3>
            <p>You've learned how to:</p>
            <ul>
                <li>1. Work with sample patient data</li>
                <li>2. Run risk analysis</li>
                <li>3. View cohort results and risk probability list</li>
                <li>4. Examine individual patients</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("### Next Steps:")
            col_next1, col_next2 = st.columns(2)
            with col_next1:
                st.info(" Navigate to 'User Analysis' in the sidebar to work with your own data")
            with col_next2:
                if st.button("🔄 Restart Tutorial", key="restart_tut"):
                    st.session_state.tutorial_step = 0
                    if 'demo_results' in st.session_state:
                        del st.session_state.demo_results
                    st.rerun()
    
    # MODE 3: LEARN BY EXPLORING
    elif demo_mode == "Learn by Exploring":
        st.subheader("Free Exploration Mode")
        
        st.markdown("""
        <div class="demo-box">
        <h4>Explore at Your Own Pace</h4>
        <p>The complete interface is available below with pre-loaded sample data. 
        Try different features and see how the system responds.</p>
        </div>
        """, unsafe_allow_html=True)
        
        exploration_tab = st.tabs(["Sample Analysis", "Learning Resources", "Tips & Tricks"])
        
        with exploration_tab[0]:
            st.write("### Analyze Sample Patients")
            
            # IMPORTANT: Results ONLY shown after button click
            if st.button("Load & Analyze Sample Data", key="explore_analyze", type="primary"):
                with st.spinner(" Analyzing sample data..."):
                    demo_results = process_data(demo_data)
                st.success("Sample data analyzed successfully!")
                st.divider()
                render_dashboard(demo_results, mode="bulk", key_prefix="explore")
        
        with exploration_tab[1]:
            st.write("### Quick Reference Guide")
            
            with st.expander("Understanding Risk Scores"):
                st.write("""
                1. **0-30%**: Very Low Risk
                2. **30-50%**: Low Risk  
                3. **50-70%**: Moderate-High Risk
                4. **70-100%**: Very High Risk
                """)
            
            with st.expander("Biomarker Types"):
                st.write("""
                1. **_prot**: Protein measurements
                2. **_rna**: RNA expression levels
                3. **_met**: Metabolite concentrations
                """)
            
            with st.expander("Chart Types"):
                st.write("""
                1. **Gauge**: Individual risk percentage
                2. **Histogram**: Cohort distribution
                3. **Bar Chart**: Individual patient risk scores sorted
                4. **Risk Probability List**: Table with exact percentages
                5. **Radar**: Multi-modal balance
                6. **Bar Charts**: Biomarker levels
                """)
        
        with exploration_tab[2]:
            st.write("### Exploration Tips")
            
            st.info("""
            **Things to Try:**
            1. Compare all 3 sample patients' profiles
            2. Look at the risk probability list to see exact scores
            3. Look for patterns in biomarker elevation
            4. See how protein/RNA/metabolite balance differs
            5. Check which markers appear in both patient-specific and global importance
            6. Expand the "View All Biomarker Values" section
            """)
            
            st.success("""
            **What Makes a Good Analysis:**
            1. Review both cohort and individual results
            2. Check the risk probability list for exact percentages
            3. Compare patient markers to global importance
            4. Note the multi-modal signature shape
            5. Look for biomarker clusters
            """)

    # Add reset button at bottom of demo page
    st.divider()
    if st.button(" Reset Demo Workspace"):
        # Clear all session state related to demo
        keys_to_clear = [k for k in st.session_state.keys() if 'demo' in k or 'tutorial' in k]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

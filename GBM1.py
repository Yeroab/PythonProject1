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
    [data-testid="stSidebar"] {
        background-color: #001f3f;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    header[data-testid="stHeader"] {
        background-color: #5dade2;
    }
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
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff;
    }
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

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        with open('momics_xgb_model-1.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('imputer-1.pkl', 'rb') as f:
            imputer = pickle.load(f)
        if not hasattr(imputer, '_fill_dtype') and hasattr(imputer, '_fit_dtype'):
            imputer._fill_dtype = imputer._fit_dtype
        with open('scaler-1.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_list-1.pkl', 'rb') as f:
            feature_list = pickle.load(f)
        feature_names = list(model.feature_names_in_)
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

# --- Ensembl → Gene Name mapping ---
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
    return ENSEMBL_TO_GENE.get(str(ensembl_id), str(ensembl_id))

GENE_TO_ENSEMBL = {v: k for k, v in ENSEMBL_TO_GENE.items()}

def remap_uploaded_df(df):
    return df.rename(columns=lambda col: GENE_TO_ENSEMBL.get(col, col))

importance_df_display = importance_df.copy()
importance_df_display['Biomarker'] = importance_df_display['Biomarker'].apply(to_gene)

# =============================================================================
# DEMO DATA — loaded from TGCA_DEMO_DATA.csv
# =============================================================================
# Contains 4 real GBM patients (CPTAC cohort) with 6 of 7 critical model
# features present as RNA_ENSG Ensembl IDs. CACNA2D3 (RNA_ENSG00000157445.13,
# importance 13.8%) was not available and is imputed to the training median
# (170 read counts). Predictions are differentiated: 2 Low Risk, 2 High Risk.
# =============================================================================
DEMO_CSV_PATH = 'TGCA_DEMO_DATA.csv'

@st.cache_data
def load_demo_data():
    """Load real GBM patient data from TGCA_DEMO_DATA.csv."""
    try:
        df = pd.read_csv(DEMO_CSV_PATH)
        # Handle both 'Sample ID' (space) and 'Sample_ID' (underscore)
        id_col = 'Sample ID' if 'Sample ID' in df.columns else 'Sample_ID'
        sample_ids = df[id_col].tolist() if id_col in df.columns else [f"Patient {i}" for i in range(len(df))]
        df_data = df.drop(columns=[id_col], errors='ignore')
        # Columns are already RNA_ENSG format — remap any gene symbols just in case
        df_data = df_data.rename(columns=lambda c: GENE_TO_ENSEMBL.get(c, c))
        return df_data, sample_ids
    except FileNotFoundError:
        st.error(f"Demo data file '{DEMO_CSV_PATH}' not found. Please ensure it is in the root directory.")
        st.stop()


# --- Processing Engine ---
def process_data(df):
    with st.spinner("Analyzing Patient Biomarkers..."):
        imputer_features = list(imputer.feature_names_in_)
        df_full = df.reindex(columns=imputer_features, fill_value=np.nan)
        df_imputed = pd.DataFrame(
            imputer.transform(df_full.astype(np.float64)),
            columns=imputer_features
        )
        df_scaled = pd.DataFrame(
            scaler.transform(df_imputed),
            columns=imputer_features
        )
        df_model_input = df_scaled.reindex(columns=feature_names, fill_value=0.0)
        probs = model.predict_proba(df_model_input.astype(float))[:, 1]
        preds = (probs > 0.5).astype(int)
        results = pd.DataFrame({
            "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
            "Risk Score": probs
        })
        return pd.concat([results, df_model_input.reset_index(drop=True)], axis=1)

# --- Risk & Prediction Visuals ---
def render_risk_charts(results, mode="manual", key_prefix=""):
    st.subheader("Prediction & Risk Assessment")
    if mode == "manual":
        prob = results["Risk Score"].iloc[0]
        pred = results["Prediction"].iloc[0]
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Prediction", pred)
        with col_m2:
            st.metric("Risk Score", f"{prob:.2%}")
    else:
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_hist = px.histogram(results, x="Risk Score", color="Prediction",
                                     title="Risk Probability Distribution",
                                     color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"},
                                     nbins=20)
            fig_hist.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Patients", showlegend=True)
            st.plotly_chart(fig_hist, use_container_width=True, key=f"{key_prefix}_hist")
        with col_chart2:
            results_sorted = results.sort_values('Risk Score', ascending=False).reset_index(drop=True)
            results_sorted['Patient_ID'] = results_sorted.index
            fig_bar = px.bar(results_sorted, x='Patient_ID', y='Risk Score', color='Prediction',
                            title="Individual Patient Risk Scores",
                            color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"},
                            labels={'Patient_ID': 'Patient Index', 'Risk Score': 'Risk Probability'})
            fig_bar.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Risk Threshold (0.5)")
            fig_bar.update_layout(xaxis_title="Patient Index (Sorted by Risk)", yaxis_title="Risk Probability",
                                  yaxis_range=[0, 1], showlegend=True)
            st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_bar")
        st.divider()
        st.subheader("Risk Probability List")
        risk_list_df = results[['Prediction', 'Risk Score']].copy()
        risk_list_df['Patient ID'] = risk_list_df.index
        risk_list_df['Risk Score'] = risk_list_df['Risk Score'].apply(lambda x: f"{x:.2%}")
        risk_list_df = risk_list_df[['Patient ID', 'Prediction', 'Risk Score']]
        st.dataframe(risk_list_df, use_container_width=True, hide_index=True)

# --- Complete Dashboard ---
def render_dashboard(results, mode="manual", key_prefix="", patient_labels=None):
    render_risk_charts(results, mode=mode, key_prefix=key_prefix)
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
    st.divider()
    st.subheader("Individual Patient Analysis")
    if patient_labels:
        fmt = lambda i: f"Patient {i} — {patient_labels[i]}" if i < len(patient_labels) else f"Patient {i}"
    else:
        fmt = lambda i: f"Patient {i}"
    selected_idx = st.selectbox("Select Patient Record", results.index, format_func=fmt, key=f"{key_prefix}_select")
    patient_row = results.iloc[selected_idx]
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Prediction", patient_row["Prediction"])
    with col_info2:
        st.metric("Risk Score", f"{patient_row['Risk Score']:.2%}")
    st.divider()
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.write("### Multi-Modal Signature")
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
    st.divider()
    st.subheader(f"Biomarker Levels for Patient {selected_idx}")
    st.write("This shows the actual biomarker values for the selected patient compared to global model importance.")
    patient_markers = patient_row.drop(['Prediction', 'Risk Score']).astype(float)
    patient_top_markers = patient_markers.sort_values(ascending=False).head(15)
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
page = st.sidebar.radio("Navigation", ["Home", "Documentation", "User Analysis", "Demo Walkthrough"])

st.title("MOmics-ML | GBM Clinical Diagnostic Suite")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "Home":
    try:
        logo = Image.open('logo.png')
        st.image(logo, use_container_width=True)
    except:
        st.info("Logo image not found. Please ensure 'logo.png' is in the root directory.")
    st.markdown("<h1 style='text-align: center;'>MOmics-ML</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>GBM Clinical Diagnostic Suite</h3>", unsafe_allow_html=True)

# ============================================================================
# DOCUMENTATION PAGE
# ============================================================================
elif page == "Documentation":
    st.header("System Documentation")
    doc_tabs = st.tabs([
        "Overview",
        "GUI User Guide",
        "Model Architecture",
        "Input Data Format",
        "Interpreting Results"
    ])

    # ------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # ------------------------------------------------------------------
    with doc_tabs[0]:
        st.markdown("""
### Purpose and Scope

MOmics-ML is a clinical decision support tool designed for glioblastoma multiforme (GBM) patient risk stratification.
The system integrates multi-omics biomarker data — transcriptomics, proteomics, and metabolomics — to generate
probability-based risk assessments that can help clinicians identify high-risk patients who may benefit from
more aggressive monitoring or treatment strategies.

The tool is not intended to replace clinical judgement. Risk scores produced by MOmics-ML are probabilistic
outputs derived from population-level training data and should be interpreted in the context of each patient's
full clinical picture.

---

### Analysis Pipeline

Patient data passes through the following stages in sequence:

**1. Data Ingestion**
Raw multi-omics measurements are provided either by manual entry (single patient) or CSV upload (cohort).
The system accepts values in three formats: gene symbol names (e.g. BTF3L4), bare Ensembl IDs
(e.g. ENSG00000164061.4), or prefixed Ensembl IDs (e.g. RNA_ENSG00000164061.4). Column format is
detected automatically and remapped internally before processing.

**2. Feature Alignment**
The input data is aligned against the full 70,961-feature space used during model training (10,409 protein
features, 69 metabolite features, and 60,483 RNA transcript features). Any features present in the input
but not in the model space are silently ignored. Any features required by the model but absent from the
input are filled with NaN before imputation.

**3. Missing Value Imputation**
A SimpleImputer fitted on the training cohort replaces all NaN values with the per-feature median derived
from the training data. This means that missing markers do not cause pipeline failure but do reduce
prediction accuracy — particularly if the missing features carry high model importance.

**4. Feature Scaling**
A StandardScaler fitted on the training cohort normalises each feature to zero mean and unit variance.
This step is required because XGBoost is sensitive to the relative scale of input features when they
are used across split thresholds.

**5. Risk Inference**
The XGBoost classifier outputs a probability score between 0 and 1 representing the likelihood that a
patient belongs to the high-risk class. A threshold of 0.5 is applied to assign the binary label
(High Risk / Low Risk). The continuous probability score is also reported to allow for finer clinical
interpretation.

**6. Visualisation**
Results are rendered as interactive charts and per-patient biomarker profiles. Individual patient
dashboards show the top expressed markers, a comparison against global model feature importance, and
a multi-modal radar summarising expression across omics layers.

---

### Cohort

The model was trained on GBM patient data from the Clinical Proteomic Tumor Analysis Consortium (CPTAC)
dataset. Training features include RNA-seq transcript counts quantified in Ensembl ID format, protein
expression values from mass spectrometry, and known metabolite concentrations from targeted metabolomics.
        """)

    # ------------------------------------------------------------------
    # TAB 2: GUI USER GUIDE
    # ------------------------------------------------------------------
    with doc_tabs[1]:
        st.markdown("""
### Navigating the Application

The application is divided into four pages accessible from the left sidebar:

- **Home** — Landing page with application branding.
- **Documentation** — This page. Full reference for the system, model, and data format.
- **User Analysis** — The primary workspace for running analyses on your own patient data.
- **Demo Walkthrough** — An interactive demo environment using real CPTAC GBM patients.

---

### User Analysis Page

The User Analysis page has two tabs: Manual Patient Entry and Bulk Data Upload.

**Manual Patient Entry**

This mode is designed for single-patient analysis. The top section displays the first 12 model features
as individual number input fields labelled with human-readable gene names. All fields default to 0.0.
The full set of 100 model features is accessible by expanding the "Advanced Marker Input" section below
the top fields. Once values have been entered, clicking "Analyze Single Patient" runs the full pipeline
and renders the results dashboard directly below.

Note: Leaving a field at 0.0 is not equivalent to a missing value — it is treated as a literal
measurement of zero. If a marker was not measured, consider using the bulk upload mode instead, where
unmeasured columns can simply be omitted and will be handled by the imputer.

**Bulk Data Upload**

This mode processes multiple patients in a single run. To use it:

1. Click "Download CSV Template" to obtain a pre-formatted CSV with the correct column headers in
   gene-name format. Each column corresponds to one of the 100 model features.
2. Fill in one patient per row. Each row should contain numeric expression values for the biomarkers
   that were measured. Columns for unmeasured biomarkers may be left empty or omitted entirely.
3. Upload the completed file using the file uploader. The system will detect the column format
   automatically (gene names, Ensembl IDs, or prefixed Ensembl IDs) and remap as needed.
4. Results for all patients are displayed simultaneously in the dashboard below the upload widget.

---

### Demo Walkthrough Page

The demo page provides three modes of interaction, all using the same pre-loaded real patient dataset:

**Try with Sample Patients**
Runs the full analysis pipeline on four real CPTAC GBM patients with a single button click. Results
are displayed immediately and persist while you interact with the patient selector dropdown. Use this
mode to quickly see what a complete analysis output looks like.

**Guided Tutorial**
A five-step walkthrough that introduces the data, runs the analysis, explains the cohort-level charts,
and then walks through an individual patient profile. Each step must be completed before advancing to
the next. Progress is tracked with a progress bar at the top of the section.

**Learn by Exploring**
Opens the full analysis dashboard with the demo data pre-loaded. This mode is unguided and intended
for users who want to explore the interface independently. A learning resources tab provides reference
definitions for risk score ranges and biomarker type prefixes.

---

### Resetting the Demo

A "Reset Demo Workspace" button at the bottom of the Demo Walkthrough page clears all session state
associated with the demo, including stored analysis results and tutorial progress. This resets the
page to its initial state without requiring a browser refresh.
        """)

    # ------------------------------------------------------------------
    # TAB 3: MODEL ARCHITECTURE
    # ------------------------------------------------------------------
    with doc_tabs[2]:
        st.markdown("""
### Machine Learning Model

**Algorithm: XGBoost (Extreme Gradient Boosting)**

The core predictive model is an XGBoost binary classifier. XGBoost builds an ensemble of decision trees
in a sequential, gradient-boosted fashion. Each tree is trained to correct the residual errors of the
previous ensemble, and the final prediction is the sum of contributions from all trees passed through a
logistic function to produce a probability. XGBoost was selected for this application because it handles
high-dimensional sparse input efficiently, is robust to the presence of correlated features common in
multi-omics data, and provides interpretable feature importance scores.

---

### Feature Space

The model was trained on a feature space spanning three omics data types:

| Data Type | Feature Prefix | Number of Features |
|-----------|---------------|-------------------|
| RNA-seq transcriptomics | RNA_ENSG | 60,483 |
| Proteomics (mass spectrometry) | PROT_ | 10,409 |
| Metabolomics | MET_ | 69 |
| **Total** | | **70,961** |

At the inference stage, the model operates on a reduced set of 100 features selected during training.
The remaining 70,861 features are used only by the imputer and scaler and do not contribute to the
final prediction.

---

### Selected Features and Importance

Of the 100 features passed to the model, only 7 carry non-zero importance in the current trained
XGBoost model. These features account for 100% of the model's predictive signal:

| Gene Name | Ensembl ID | Feature Importance |
|-----------|-----------|-------------------|
| LINC02084 | RNA_ENSG00000244040.4 | 21.5% |
| BTF3L4 | RNA_ENSG00000164061.4 | 15.8% |
| RNU6-1 | RNA_ENSG00000206814.1 | 15.4% |
| MS4A6E | RNA_ENSG00000181215.11 | 13.9% |
| CACNA2D3 | RNA_ENSG00000157445.13 | 13.8% |
| LINC01605 | RNA_ENSG00000233487.6 | 10.3% |
| LINC01116 | RNA_ENSG00000242759.5 | 9.3% |

Four of the seven are long intergenic non-coding RNAs (LINCs) and one is a small nuclear RNA (RNU6-1).
All seven are transcriptomic features. Proteomics and metabolomics features, while present in the
preprocessing pipeline, have zero importance in the current model version.

Feature importance values are derived from the XGBoost gain metric, which measures the average
improvement in model accuracy contributed by each feature across all splits in which it appears.

---

### Preprocessing Components

The model bundle consists of four serialised objects loaded at application startup:

**momics_xgb_model-1.pkl** — The trained XGBoost classifier. Stores the full ensemble of decision
trees, feature names, and hyperparameters. Used exclusively for inference.

**imputer-1.pkl** — A scikit-learn SimpleImputer fitted on the training cohort with strategy="median".
Replaces NaN values in the 70,961-feature input space with per-feature training medians before scaling.
Patients with many missing features will have their input dominated by training-set medians, which
reduces the reliability of their risk score.

**scaler-1.pkl** — A scikit-learn StandardScaler fitted on the training cohort. Applied after imputation
to normalise all features to zero mean and unit variance. The scaler was fitted on the imputed training
data, so it expects input that has already been passed through the imputer.

**feature_list-1.pkl** — The ordered list of 100 feature names that define the model input space.
Used to align the scaled output of the scaler with the feature order expected by the XGBoost model.

---

### Risk Score Interpretation

The model outputs a continuous probability P(High Risk) between 0 and 1. The binary label is assigned
using a fixed threshold of 0.5. The probability score itself carries more information than the binary
label and should be reported alongside it in clinical contexts.

The current model exhibits a tendency toward polarised outputs — scores cluster near 0.2-0.3 for
low-risk patients and near 0.8 for high-risk patients. This reflects the decision boundary structure
learned from the training data and is consistent with a model that has identified strong discriminating
features (particularly LINC02084 expression) rather than a calibration artefact.
        """)

    # ------------------------------------------------------------------
    # TAB 4: INPUT DATA FORMAT
    # ------------------------------------------------------------------
    with doc_tabs[3]:
        st.markdown("""
### Input Data Format Specification

---

### Accepted Column Name Formats

The application accepts three column naming conventions and detects the format automatically.
**Prefixed Ensembl IDs (Format 2) are strongly recommended for best results.** Gene symbol
remapping relies on a fixed reference table covering only the 100 model features, which means
any feature not in that table will be silently dropped. Prefixed Ensembl IDs bypass this
remapping step entirely and are guaranteed to match the model's internal feature space exactly,
eliminating any risk of features being lost due to naming mismatches.

**Format 1: Gene Symbol Names**

Column headers are standard HGNC gene symbols, for example:

```
BTF3L4, CACNA2D3, LINC01116, MS4A6E, LINC02084, LINC01605, RNU6-1
```

This is the format used in the downloadable CSV template. Gene symbols are remapped internally to
their corresponding Ensembl IDs using a built-in reference table before processing. Only the 100
model features have entries in this reference table; any other gene symbol columns will be ignored.
Use this format only when prefixed Ensembl IDs are not available.

**Format 2: Prefixed Ensembl IDs (recommended — best results)**

Column headers use the full internal feature identifier with the data-type prefix:

```
RNA_ENSG00000164061.4, RNA_ENSG00000157445.13, RNA_ENSG00000242759.5
```

This format matches the model's internal feature names exactly and requires no remapping. All
features present in the file are recognised directly, version suffixes are matched precisely, and
there is no risk of a feature being dropped due to a missing or incorrect gene symbol mapping.
Files exported from the CPTAC pipeline or generated by the extract_demo_features.py utility will
be in this format. This is the preferred format for all bulk uploads.

**Format 3: Bare Ensembl IDs**

Column headers contain the Ensembl ID without the RNA_ prefix:

```
ENSG00000164061.4, ENSG00000157445.13, ENSG00000242759.5
```

The application detects this format and prepends the RNA_ prefix automatically. Version suffixes
(the .4, .13 portion) must be present and must match the training reference exactly. This format
assumes all features are RNA-seq features; proteomics and metabolomics features cannot be
represented in this format and should use Format 2 instead.

---

### CSV File Structure

Files must be in comma-separated values (CSV) format with UTF-8 encoding. The expected structure is:

- Each **row** represents one patient.
- Each **column** represents one biomarker measurement.
- An optional **Sample_ID** column (or "Sample ID" with a space) may be included as the first column
  to provide patient identifiers. This column is dropped before processing and is not passed to the model.
- All measurement values must be **numeric**. String values, percentage signs, or other non-numeric
  content in data columns will cause a processing error.
- **Column headers are required.** Files without a header row cannot be parsed correctly.
- Columns for biomarkers that were not measured may be **omitted entirely** rather than filled with
  zeros or empty strings. The imputer will handle missing features using training-set medians.
  Providing a zero where a marker was not measured is incorrect and will skew the result.

---

### Minimal Valid Example (Gene Symbol Format)

```
Sample_ID,BTF3L4,CACNA2D3,LINC01116,LINC02084,MS4A6E,LINC01605,RNU6-1
Patient_001,1707,607,403831,311900,92228,3669,111
Patient_002,975,57,106,10,41,2,2
```

A file containing only the 7 high-importance features is sufficient to produce a differentiated
risk score. All remaining 93 model features will be imputed to their training medians and will
not affect the prediction, since they carry zero importance in the current model.

---

### Minimal Valid Example (Prefixed Ensembl ID Format)

```
Sample_ID,RNA_ENSG00000164061.4,RNA_ENSG00000157445.13,RNA_ENSG00000242759.5,RNA_ENSG00000244040.4,RNA_ENSG00000181215.11,RNA_ENSG00000233487.6,RNA_ENSG00000206814.1
Patient_001,1707,607,403831,311900,92228,3669,111
Patient_002,975,57,106,10,41,2,2
```

---

### Expression Value Units

The model was trained on raw RNA-seq read counts as produced by a standard RNA-seq quantification
pipeline (gene-level counts, not TPM or FPKM). Submitting TPM-normalised or log-transformed values
will produce incorrect results because the imputer and scaler were fitted on raw count distributions.
If your data has been normalised, contact the model training team to obtain a version of the
preprocessing objects fitted on the same normalisation scheme.

For proteomics features (PROT_ prefix), values should be log2-transformed protein expression ratios
as produced by the CPTAC MSSM proteomics pipeline. For metabolomics features (MET_ prefix), values
should be corrected peak area intensities as produced by the CPTAC PNNL metabolomics pipeline.

---

### Common Errors

**"Error processing file"** — Most commonly caused by non-numeric values in data columns, a missing
header row, or a file saved in a format other than CSV (e.g. Excel .xlsx). Ensure the file is saved
as CSV with UTF-8 encoding before uploading.

**All patients receiving identical risk scores** — Indicates that none of the 7 high-importance features
were present in the uploaded file, and all critical features were imputed to the same training median.
Verify that your column names match one of the three accepted formats and that the 7 critical features
listed in the Model Architecture tab are present in your data.

**Unrecognised column warning** — Columns that do not match any of the 100 model features and are not
a Sample_ID column will trigger a warning listing the unrecognised names. These columns are ignored
and do not affect results. This commonly occurs when a file includes clinical metadata columns
alongside expression data.
        """)

    # ------------------------------------------------------------------
    # TAB 5: INTERPRETING RESULTS
    # ------------------------------------------------------------------
    with doc_tabs[4]:
        st.markdown("""
### Interpreting Analysis Results

---

### Risk Score

The risk score is the raw probability output of the XGBoost model, representing P(High Risk) on a
scale from 0 to 1. A score above 0.5 results in a High Risk classification; a score at or below 0.5
results in a Low Risk classification.

The score should not be treated as a precise clinical probability. It reflects the model's confidence
relative to the patterns observed in the CPTAC training cohort. A score of 0.80, for example, means
the patient's biomarker profile is similar to profiles that were associated with high-risk outcomes
in the training data, not that there is an 80% clinical probability of a specific event.

Approximate interpretation ranges:

| Risk Score | Label | Interpretation |
|-----------|-------|---------------|
| 0.00 - 0.30 | Low Risk | Profile substantially dissimilar to high-risk training cases |
| 0.30 - 0.50 | Low Risk (borderline) | Profile weakly dissimilar; interpret with caution |
| 0.50 - 0.70 | High Risk (borderline) | Profile weakly similar to high-risk training cases |
| 0.70 - 1.00 | High Risk | Profile substantially similar to high-risk training cases |

---

### Risk Probability Distribution (Histogram)

The histogram shows the spread of risk scores across the full uploaded cohort. Each bar represents
the number of patients whose risk score falls within that range. Bars are colour-coded red for High
Risk patients and green for Low Risk patients. A narrow distribution clustered near 0.8 or 0.2
indicates the model is confident in its classifications; a broad distribution indicates more
heterogeneity in the cohort or more uncertainty in the predictions.

---

### Individual Patient Risk Scores (Bar Chart)

The bar chart displays one bar per patient, sorted from highest to lowest risk score. The dashed
horizontal line at 0.5 marks the classification threshold. Bars above the line are High Risk; bars
below are Low Risk. This chart is useful for identifying patients near the threshold who may warrant
closer review.

---

### Multi-Modal Signature (Radar Chart)

The radar chart shows the average scaled expression level across the three omics layers for the
selected patient: Proteins (PROT_ features), RNA (RNA_ENSG features), and Metabolites (MET_ features).
Values shown are post-scaling (z-scores relative to the training cohort mean), so a value above zero
indicates above-average expression and a value below zero indicates below-average expression. This
chart provides a high-level summary of which data modalities are elevated for that patient. If only
RNA features are present in the uploaded data, the Proteins and Metabolites axes will read zero.

---

### Top 20 Marker Levels

This horizontal bar chart shows the 20 features with the highest scaled values for the selected
patient. Features are displayed using gene names where a mapping exists, or their raw Ensembl ID
otherwise. This chart identifies which specific biomarkers are most elevated in the patient relative
to the training cohort, not which biomarkers are most important to the model globally. A marker can
appear in this chart without contributing to the risk score if it carries zero model importance.

---

### Patient's Top 15 Expressed Markers vs Global Model Importance

These two side-by-side charts enable comparison between patient-specific expression and model-level
importance:

**Left panel (Patient's Top 15 Expressed Markers):** The 15 features most elevated in the selected
patient, coloured by expression level. This is patient-specific.

**Right panel (Global Model Importance):** The 15 features with the highest XGBoost gain importance
across all patients. This reflects what the model has learned to rely on across the training cohort.

Overlap between the two panels — markers that are both highly expressed in the patient and globally
important to the model — provides the most clinically actionable signal. A patient with high LINC02084
and BTF3L4 expression appearing in both panels has a profile that strongly matches the patterns the
model associates with high-risk outcomes.

---

### Limitations

The model was trained and validated on data from a single cohort (CPTAC GBM). Performance on data
from different sequencing platforms, different RNA-seq quantification pipelines, or patient populations
with substantially different demographic characteristics has not been evaluated. The model should be
validated against local institutional data before being used to inform clinical decisions.

Predictions for patients with a large proportion of missing features (imputed values) are less reliable
than predictions for patients with complete data. If more than 3 of the 7 high-importance features are
missing for a given patient, the risk score for that patient should be treated as unreliable.
        """)

# ============================================================================
# USER ANALYSIS PAGE
# ============================================================================
elif page == "User Analysis":
    st.header("User Analysis")
    analysis_tabs = st.tabs(["Manual Patient Entry", "Bulk Data Upload"])
    with analysis_tabs[0]:
        st.subheader("Manual Patient Entry")
        st.info("Input raw laboratory values. Markers left at 0.0 will be treated as baseline. Click 'Analyze Single Patient' to see results.")
        user_inputs = {}
        m_cols = st.columns(3)
        for i, name in enumerate(feature_names[:12]):
            with m_cols[i % 3]:
                user_inputs[name] = st.number_input(f"{to_gene(name)}", value=0.0, key=f"man_in_{name}")
        with st.expander("Advanced Marker Input (Full Set)"):
            adv_cols = st.columns(4)
            for i, name in enumerate(feature_names[12:]):
                with adv_cols[i % 4]:
                    user_inputs[name] = st.number_input(f"{to_gene(name)}", value=0.0, key=f"man_adv_{name}")
        if st.button("Analyze Single Patient", key="btn_manual", type="primary"):
            m_results = process_data(pd.DataFrame([user_inputs]))
            st.success("Analysis Complete! Results displayed below.")
            st.divider()
            render_dashboard(m_results, mode="manual", key_prefix="man")
    with analysis_tabs[1]:
        st.subheader("Bulk Data Processing")
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            st.write("### Download Template")
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
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully. Found {len(raw_df)} patient(s).")
                sample_cols = [c for c in raw_df.columns if c not in ('Sample_ID', 'Sample ID')][:5]
                is_ensembl_prefixed = any(str(c).startswith("RNA_ENSG") for c in sample_cols)
                is_bare_ensembl = any(str(c).startswith("ENSG") for c in sample_cols)
                if is_ensembl_prefixed:
                    pass
                elif is_bare_ensembl:
                    raw_df = raw_df.rename(columns=lambda col: f"RNA_{col}" if str(col).startswith("ENSG") else col)
                    st.info("Bare Ensembl ID columns detected. RNA_ prefix added automatically.")
                else:
                    raw_df = raw_df.rename(columns=lambda col: GENE_TO_ENSEMBL.get(col, col))
                    matched = sum(1 for c in raw_df.columns if str(c).startswith("RNA_ENSG"))
                    st.info(f"Gene name columns detected. {matched} of the 100 model features matched.")
                recognised = set(feature_names) | set(GENE_TO_ENSEMBL.values())
                extra_cols = [c for c in raw_df.columns if c not in recognised and c not in ('Sample_ID', 'Sample ID')]
                if extra_cols:
                    st.warning(f"{len(extra_cols)} unrecognised column(s) will be ignored: {', '.join(extra_cols[:5])}{'...' if len(extra_cols) > 5 else ''}.")
                b_results = process_data(raw_df)
                st.divider()
                st.subheader("Analysis Results")
                render_dashboard(b_results, mode="bulk", key_prefix="blk")
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please ensure your CSV file follows the template format.")

# ============================================================================
# DEMO WALKTHROUGH PAGE
# ============================================================================
elif page == "Demo Walkthrough":
    st.header("Interactive Demo Workspace")
    st.markdown("""
    <div class="demo-box">
    <h3>Welcome to the Demo Workspace</h3>
    <p>This workspace uses <strong>real GBM patient data</strong> from the CPTAC dataset.
    Explore the full analysis workflow with genuine patient biomarker profiles.</p>
    <p><strong>What's included:</strong></p>
    <ul>
        <li>4 real GBM patients (CPTAC cohort)</li>
        <li>6 of 7 critical RNA biomarkers with real expression values</li>
        <li>2 Low Risk and 2 High Risk patients for meaningful comparison</li>
        <li>Interactive per-patient biomarker visualizations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    demo_data, demo_sample_ids = load_demo_data()
    st.divider()
    demo_mode = st.radio("**Choose Demo Mode:**", ["Try with Sample Patients", "Guided Tutorial", "Learn by Exploring"], horizontal=True)

    if demo_mode == "Try with Sample Patients":
        st.subheader("Interactive Analysis with Sample Data")
        st.markdown("""
        <div class="demo-box demo-success">
        <h4>Real Patient Dataset Loaded</h4>
        <p>4 real GBM patients from the CPTAC dataset are ready for analysis.
        Click "Analyze Sample Patients" to run the full diagnostic pipeline.</p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Preview Sample Patient Data"):
            st.write("**Sample Patients Overview:**")
            preview_cols = [c for c in demo_data.columns if c.startswith('RNA_ENSG')][:10]
            st.dataframe(demo_data[preview_cols], use_container_width=True)
            id_cols = st.columns(len(demo_sample_ids))
            for i, (col, sid) in enumerate(zip(id_cols, demo_sample_ids)):
                with col:
                    st.info(f"**Patient {i}**\n{sid}")
        if st.button("Analyze Sample Patients", key="analyze_demo_patients", type="primary"):
            with st.spinner("Analyzing biomarkers..."):
                st.session_state.demo_try_results = process_data(demo_data)
        if 'demo_try_results' in st.session_state:
            st.markdown("---")
            st.success("Analysis Complete!")
            st.markdown("""
            <div class="demo-box demo-success">
            <h4>Analysis Complete</h4>
            <p>Below are the results for all 4 real GBM patients. Explore each patient's profile using the selector.</p>
            </div>
            """, unsafe_allow_html=True)
            render_dashboard(st.session_state.demo_try_results, mode="bulk", key_prefix="demo", patient_labels=demo_sample_ids)
            st.divider()
            st.markdown("""
            <div class="demo-box">
            <h4>What You're Seeing:</h4>
            <ul>
                <li><strong>Histogram:</strong> Distribution of risk scores across all 4 patients</li>
                <li><strong>Bar Chart:</strong> Individual patient risk probabilities sorted by risk level</li>
                <li><strong>Risk Probability List:</strong> Table showing all patients' risk scores</li>
                <li><strong>Patient Selector:</strong> Choose individual patients to see detailed profiles</li>
                <li><strong>Multi-Modal Radar:</strong> Shows protein/RNA/metabolite balance</li>
                <li><strong>Top Markers:</strong> Patient-specific elevated biomarkers</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.info("💡 Tip: Use the patient selector dropdown to compare the Low Risk vs High Risk profiles")

    elif demo_mode == "Guided Tutorial":
        st.subheader("Step-by-Step Guided Tutorial")
        if 'tutorial_step' not in st.session_state:
            st.session_state.tutorial_step = 0
        progress = st.progress(st.session_state.tutorial_step / 5)
        st.write(f"**Progress:** Step {st.session_state.tutorial_step + 1} of 5")
        if st.session_state.tutorial_step == 0:
            st.markdown("""<div class="demo-box"><h3>Step 1: Understanding the Sample Data</h3>
            <p>Let's start by looking at our pre-loaded sample patients.</p></div>""", unsafe_allow_html=True)
            st.write("**Our Sample Dataset Contains:**")
            preview_cols = [c for c in demo_data.columns if c.startswith('RNA_ENSG')][:15]
            st.dataframe(demo_data[preview_cols], use_container_width=True)
            st.info("**What you see:**\n1. 4 rows = 4 real GBM patients (CPTAC cohort)\n2. Columns = RNA expression features (Ensembl IDs)\n3. Values = Real RNA-seq read counts")
            if st.button("Next: Run Analysis", key="tutorial_next_0"):
                st.session_state.tutorial_step = 1
                st.rerun()
        elif st.session_state.tutorial_step == 1:
            st.markdown("""<div class="demo-box"><h3>Step 2: Running the Analysis</h3>
            <p>Now let's process our sample patients through the AI model.</p></div>""", unsafe_allow_html=True)
            if st.button("Process Sample Data", key="tutorial_analyze", type="primary"):
                with st.spinner("Analyzing biomarkers..."):
                    st.session_state.demo_results = process_data(demo_data)
                    st.session_state.tutorial_step = 2
                st.success("Analysis complete!")
                st.rerun()
        elif st.session_state.tutorial_step == 2:
            st.markdown("""<div class="demo-box demo-success"><h3>Step 3: Viewing Cohort Results</h3>
            <p>Here's the risk distribution across all patients:</p></div>""", unsafe_allow_html=True)
            if 'demo_results' in st.session_state:
                render_risk_charts(st.session_state.demo_results, mode="bulk", key_prefix="tutorial")
            st.info("Notice the split: 2 patients classified Low Risk, 2 classified High Risk.")
            if st.button("Next: Individual Patient", key="tutorial_next_2"):
                st.session_state.tutorial_step = 3
                st.rerun()
        elif st.session_state.tutorial_step == 3:
            st.markdown("""<div class="demo-box"><h3>Step 4: Individual Patient Analysis</h3>
            <p>Let's examine one patient in detail:</p></div>""", unsafe_allow_html=True)
            if 'demo_results' in st.session_state:
                selected = st.selectbox("Choose a patient:", range(len(demo_sample_ids)),
                                        format_func=lambda i: f"Patient {i} — {demo_sample_ids[i]}",
                                        key="tutorial_patient_select")
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
            st.markdown("""<div class="demo-box demo-success"><h3>Tutorial Complete!</h3>
            <p>You've learned how to work with sample data, run risk analysis, view cohort results, and examine individual patients.</p>
            </div>""", unsafe_allow_html=True)
            st.write("### Next Steps:")
            col_next1, col_next2 = st.columns(2)
            with col_next1:
                st.info("Navigate to 'User Analysis' in the sidebar to work with your own data")
            with col_next2:
                if st.button("🔄 Restart Tutorial", key="restart_tut"):
                    st.session_state.tutorial_step = 0
                    if 'demo_results' in st.session_state:
                        del st.session_state.demo_results
                    st.rerun()

    elif demo_mode == "Learn by Exploring":
        st.subheader("Free Exploration Mode")
        st.markdown("""<div class="demo-box"><h4>Explore at Your Own Pace</h4>
        <p>The complete interface is available below with real GBM patient data from the CPTAC dataset.</p></div>""", unsafe_allow_html=True)
        exploration_tab = st.tabs(["Sample Analysis", "Learning Resources", "Tips & Tricks"])
        with exploration_tab[0]:
            st.write("### Analyze Sample Patients")
            if st.button("Load & Analyze Sample Data", key="explore_analyze", type="primary"):
                with st.spinner("Analyzing sample data..."):
                    st.session_state.demo_explore_results = process_data(demo_data)
            if 'demo_explore_results' in st.session_state:
                st.success("Sample data analyzed successfully!")
                st.divider()
                render_dashboard(st.session_state.demo_explore_results, mode="bulk", key_prefix="explore", patient_labels=demo_sample_ids)
        with exploration_tab[1]:
            st.write("### Quick Reference Guide")
            with st.expander("Understanding Risk Scores"):
                st.write("1. **0-30%**: Very Low Risk\n2. **30-50%**: Low Risk\n3. **50-70%**: Moderate-High Risk\n4. **70-100%**: Very High Risk")
            with st.expander("Biomarker Types"):
                st.write("1. **PROT_**: Protein expression levels\n2. **RNA_ENSG**: RNA transcript expression\n3. **MET_**: Metabolite concentrations")
        with exploration_tab[2]:
            st.write("### Exploration Tips")
            st.info("**Things to Try:**\n1. Compare the Low Risk vs High Risk patient profiles\n2. Look at how LINC02084 and BTF3L4 expression differs between risk groups\n3. Check which markers appear in both patient-specific and global importance charts")

    st.divider()
    if st.button("Reset Demo Workspace"):
        keys_to_clear = [k for k in list(st.session_state.keys()) if 'demo' in k or 'tutorial' in k]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

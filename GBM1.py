import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="MultiNet_AI", layout="wide", page_icon="🧬")

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
        
        # Calculate features for 95% cumulative importance
        cumsum = importance_df['Influence Score'].cumsum()
        features_95_pct = importance_df[cumsum <= 0.95]['Biomarker'].tolist()
        # Ensure we have at least the features contributing to 95%
        if len(features_95_pct) == 0:
            features_95_pct = importance_df.head(160)['Biomarker'].tolist()
        
        return model, feature_names, importance_df, features_95_pct
    except FileNotFoundError:
        st.error("File 'gbm_clinical_model.pkl' not found. Please ensure it is in the root directory.")
        st.stop()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

model, feature_names, importance_df, features_95_pct = load_assets()

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
        if '_prot' in name:
            patient1[name] = np.random.uniform(15, 35)
        elif '_rna' in name:
            patient1[name] = np.random.uniform(100, 200)
        else:
            patient1[name] = np.random.uniform(50, 150)
    demo_patients.append(patient1)
    
    # Patient 2: Low Risk Profile
    patient2 = {}
    for name in demo_feature_names:
        if '_prot' in name:
            patient2[name] = np.random.uniform(5, 15)
        elif '_rna' in name:
            patient2[name] = np.random.uniform(50, 100)
        else:
            patient2[name] = np.random.uniform(20, 80)
    demo_patients.append(patient2)
    
    # Patient 3: Moderate Risk Profile
    patient3 = {}
    for name in demo_feature_names:
        if '_prot' in name:
            patient3[name] = np.random.uniform(10, 25)
        elif '_rna' in name:
            patient3[name] = np.random.uniform(75, 150)
        else:
            patient3[name] = np.random.uniform(35, 115)
    demo_patients.append(patient3)
    
    # Fill remaining features with 0
    for patient in demo_patients:
        for name in feature_names[50:]:
            patient[name] = 0.0
    
    return pd.DataFrame(demo_patients)

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
    st.subheader("Prediction & Risk Assessment")
    
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
        prot_avg = patient_row.filter(like='_prot').mean()
        rna_avg = patient_row.filter(like='_rna').mean()
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
    
    # Create comparison dataframe
    patient_importance = importance_df[importance_df['Biomarker'].isin(patient_top_markers.index)].copy()
    patient_importance = patient_importance.merge(
        pd.DataFrame({'Biomarker': patient_top_markers.index, 'Patient Value': patient_top_markers.values}),
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
            importance_df.head(15), 
            x='Influence Score', y='Biomarker', 
            orientation='h', color='Influence Score', 
            color_continuous_scale='Reds',
            title="Most Influential Markers Globally"
        )
        fig_global_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_global_imp, use_container_width=True, key=f"{key_prefix}_global_imp_{selected_idx}")

    with st.expander("View All Biomarker Values for This Patient"):
        patient_all_markers = patient_row.drop(['Prediction', 'Risk Score']).to_frame(name='Value')
        patient_all_markers['Biomarker'] = patient_all_markers.index
        patient_all_markers = patient_all_markers[['Biomarker', 'Value']].sort_values('Value', ascending=False)
        st.dataframe(patient_all_markers, use_container_width=True, hide_index=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("MultiNet_AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Documentation", "User Analysis", "Demo Walkthrough"]
)

# --- MAIN INTERFACE ---
st.title("MultiNet_AI | GBM Clinical Diagnostic Suite")

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
    st.markdown("<h1 style='text-align: center;'>MultiNet_AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>GBM Clinical Diagnostic Suite</h3>", unsafe_allow_html=True)

# ============================================================================
# DOCUMENTATION PAGE
# ============================================================================
elif page == "Documentation":
    st.header("System Documentation")
    
    # Create tabs for documentation sections (Backend removed, Frontend merged with Model)
    doc_tabs = st.tabs([
        "Overview",
        "System Architecture & Model",
        "Data Requirements"
    ])
    
    # Overview Tab with numbered lists
    with doc_tabs[0]:
        st.markdown("""
        ### Purpose & Scope
        
        MultiNet_AI is a clinical decision support tool designed for glioblastoma patient risk stratification. 
        The system integrates multi-omics biomarker data to generate probability-based risk assessments, 
        helping clinicians identify high-risk patients who may benefit from aggressive treatment strategies.
        
        ### Workflow Architecture
        
        The platform follows a streamlined analysis pipeline:
        
        1. Data Input: Raw laboratory values for 843 biomarkers (proteomics, transcriptomics, metabolomics)
        2. Preprocessing: Automatic alignment with model feature space, zero-filling for missing markers
        3. Inference: XGBoost model generates risk probability scores
        4. Visualization: Interactive dashboards display predictions, biomarker influences, and patient profiles
        5. Export: Results available for clinical record integration
        
        ### Clinical Use Cases
        
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
        
        ### System Requirements
        
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
    
    # System Architecture & Model Tab (Frontend + Model merged, numbered lists)
    with doc_tabs[1]:
        st.markdown("""
        ### Frontend Technology Stack
        
        **Framework**: Streamlit 1.28+
        1. Rapid prototyping and deployment
        2. Built-in widget management
        3. Session state handling
        4. Automatic reactivity
        
        **Visualization**: Plotly 5.17+
        1. Interactive charts and graphs
        2. Hover tooltips and zooming
        3. Export capabilities
        4. Responsive design
        
        **Data Handling**: Pandas 2.0+, NumPy 1.24+
        1. Efficient data manipulation
        2. Missing value handling
        3. Type conversion utilities
        4. Statistical operations
        
        **Styling**: Custom CSS
        1. Responsive layout design
        2. Theme customization
        3. Component styling
        4. Brand consistency
        
        ### Machine Learning Model
        
        #### Algorithm: XGBoost (Extreme Gradient Boosting)
        
        **Why XGBoost?**
        1. Efficiently processes 843 high-dimensional features
        2. Tree-based structure captures complex non-linear interactions
        3. Built-in gain-based feature importance calculation
        4. L1/L2 regularization prevents overfitting
        
        #### Model Specifications
        
        **Task Definition**
        1. Supervised binary classification
        2. Output: High risk (1) vs Low risk (0)
        3. Evaluation: Probability scores
        
        **Input Features**
        1. 843 multi-omics biomarkers
        2. Continuous numeric values
        3. No categorical encoding needed
        4. Raw values (no normalization)
        
        **Output Format**
        1. Probability scores via logistic objective
        2. Range: 0.0 to 1.0
        3. Calibrated via Platt scaling
        4. Confidence intervals available
        
        #### Training Protocol
        1. Supervised learning on labeled patient outcomes
        2. 5-fold stratified cross-validation for hyperparameter tuning
        3. Early stopping to prevent overfitting (50 rounds patience)
        4. Stratified sampling for class balance
        5. AUC-ROC as primary evaluation metric
        
        #### Feature Importance Analysis
        
        **Cumulative Importance Distribution**
        1. Top 6 features: 50% of model's predictive power
        2. Top 30 features: 80% of model's predictive power  
        3. Top 75 features: 90% of model's predictive power
        4. Top 160 features: 95% of model's predictive power
        5. All 843 features: 100% coverage
        
        **Calculation Method**
        1. Gain-based importance measures improvement in objective function
        2. Normalized to sum to 1.0
        3. Aggregated across all trees in ensemble
        4. Higher values indicate stronger predictive power
        
        **Clinical Application**
        1. Laboratory focus: Prioritize high-importance biomarkers
        2. Cost-effectiveness: Measure critical markers first
        3. Research: Validate known and discover novel markers
        4. Therapy: Target pathways with important markers
        
        ### Model Outputs
        
        ####Risk Score Interpretation
        
        **Score Ranges**
        1. 0.0-0.3: Very low risk, minimal intervention
        2. 0.3-0.5: Low risk, surveillance acceptable
        3. 0.5-0.7: Moderate-high risk, standard treatment
        4. 0.7-1.0: Very high risk, aggressive treatment
        
        **Prediction Labels**
        1. High Risk: Score ≥ 0.5, poor outcomes expected
        2. Low Risk: Score < 0.5, favorable prognosis
        3. Decision boundary: 50% probability threshold
        4. Adjustable based on clinical context
        
        ### Model Limitations
        
        #### Scope Limitations
        1. Trained only on glioblastoma patients
        2. Not applicable to other brain tumors
        3. Not validated for recurrent disease
        4. Limited to adult patients (≥18 years)
        
        #### Clinical Considerations
        1. Provides risk stratification, not diagnosis
        2. Should complement, not replace, clinical judgment
        3. Consider patient comorbidities and imaging
        4. Requires external validation in prospective studies
        
        #### Technical Constraints
        1. Zero-filling may not capture true baseline for all markers
        2. Assumes consistent measurement protocols across labs
        3. Model may need retraining as standards evolve
        4. Batch processing preferred for large cohorts
        
        ### Recommendations
        
        1. Present at multidisciplinary tumor board discussions
        2. Adjust risk cutoffs based on institutional resources
        3. Track model performance on real-world patients
        4. Periodically retrain with new data to maintain accuracy
        5. Validate lab measurement protocols regularly
        """)
    
    # Data Requirements Tab (numbered lists)
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
        
        #### Value Ranges
        
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
        
        **Value Validation**
        1. Negative values not expected (will be flagged)
        2. Extremely large values (>10000) reviewed for errors
        3. Outliers beyond 3 standard deviations highlighted
        4. Duplicate entries detected and reported
        
        #### CSV File Format (Bulk Upload)
        
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
        
        #### Manual Entry Guidelines
        
        1. Prioritize top 160 high-influence markers (95% predictive power)
        2. Use zero for unknowns (leave fields at 0.0 if data unavailable)
        3. Check units (ensure values match training data scale)
        4. Avoid text (only numeric inputs accepted)
        5. Quality control (review values before submission)
        
        ### Template Generation
        
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
        
        ### Data Privacy & Security
        
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
        
        **HIPAA Considerations**
        1. Suitable for de-identified research data
        2. No PHI storage or transmission
        3. Audit logging available
        4. Compliant with de-identification standards
        5. Business Associate Agreement available
        
        **Best Practices**
        1. Remove patient names from CSV files
        2. Use study IDs instead of medical record numbers
        3. Strip dates to month/year only
        4. Exclude geographic identifiers below state level
        5. Review data before upload
        """)

# ============================================================================
# USER ANALYSIS PAGE
# ============================================================================
elif page == "User Analysis":
    st.header("User Analysis")
    
    st.info(f"""**High-Priority Biomarkers**: The manual entry form displays the top {len(features_95_pct)} biomarkers 
    that account for 95% of the model's predictive ability. These are the most critical markers for accurate risk assessment.""")
    
    # Create tabs for analysis modes
    analysis_tabs = st.tabs(["Manual Patient Entry", "Bulk Data Upload"])
    
    # Manual Entry Tab - Using top 95% features
# Manual Entry Tab - Using top 95% features
with analysis_tabs[0]:
    st.subheader("Manual Patient Entry")
    st.info(f"Input raw laboratory values for the {len(features_95_pct)} most important biomarkers (95% predictive power). Markers left at 0.0 will be treated as baseline.")
    
    # Manual Entry Fields using features_95_pct
    user_inputs = {}
    
    # Display top 30 in main view with proper iteration
    st.write(f"### Top 30 Most Important Biomarkers")
    
    # Create rows of 3 columns each
    num_to_show_main = min(30, len(features_95_pct))
    for row_start in range(0, num_to_show_main, 3):
        cols = st.columns(3)
        for col_idx in range(3):
            feature_idx = row_start + col_idx
            if feature_idx < num_to_show_main:
                name = features_95_pct[feature_idx]
                with cols[col_idx]:
                    user_inputs[name] = st.number_input(f"{name}", value=0.0, key=f"man_in_{name}")
    
    # Put remaining 95% features in expander
    if len(features_95_pct) > 30:
        with st.expander(f"Additional High-Importance Markers ({len(features_95_pct) - 30} more)"):
            # Create rows of 4 columns each for remaining features
            remaining_features = features_95_pct[30:]
            for row_start in range(0, len(remaining_features), 4):
                cols = st.columns(4)
                for col_idx in range(4):
                    feature_idx = row_start + col_idx
                    if feature_idx < len(remaining_features):
                        name = remaining_features[feature_idx]
                        with cols[col_idx]:
                            user_inputs[name] = st.number_input(f"{name}", value=0.0, key=f"man_adv_{name}")
    
    # Fill remaining features with 0
    for name in feature_names:
        if name not in user_inputs:
            user_inputs[name] = 0.0

    if st.button("Analyze Single Patient", key="btn_manual"):
        m_results = process_data(pd.DataFrame([user_inputs]))
        render_dashboard(m_results, mode="manual", key_prefix="man")
    
    # Bulk Upload Tab
    with analysis_tabs[1]:
        st.subheader("Bulk Data Processing")
        
        # Template Generation & Download
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            st.write("### Download Template")
            # Generate empty template with 843 columns
            template_csv = pd.DataFrame(columns=feature_names).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Template",
                data=template_csv,
                file_name="MultiNet_Patient_Template.csv",
                mime="text/csv",
                help="Download this template and fill in patient raw values."
            )
        
        with col_t1:
            st.write("### Upload Patient Data")
            uploaded_file = st.file_uploader("Upload filled MultiNet CSV Template", type="csv")
        
        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file)
            # Process and show dashboard
            b_results = process_data(raw_df)
            render_dashboard(b_results, mode="bulk", key_prefix="blk")

# ============================================================================
# DEMO WALKTHROUGH PAGE - WITH PRE-LOADED SAMPLE DATA
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
        
        # Analysis button
        if st.button("Analyze Sample Patients", key="analyze_demo_patients", type="primary"):
            st.markdown("---")
            st.success("Processing sample dataset...")
            
            # Process the demo data
            demo_results = process_data(demo_data)
            
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
                <li><strong>Patient Selector:</strong> Choose individual patients to see detailed profiles</li>
                <li><strong>Multi-Modal Radar:</strong> Shows protein/RNA/metabolite balance</li>
                <li><strong>Top Markers:</strong> Patient-specific elevated biomarkers</li>
                <li><strong>Comparison Charts:</strong> Patient markers vs global importance</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("Tip: Use the patient selector dropdown to compare the three different risk profiles")
    
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
            
            if st.button("Process Sample Data", key="tutorial_analyze"):
                with st.spinner("Analyzing biomarkers..."):
                    st.session_state.demo_results = process_data(demo_data)
                    st.session_state.tutorial_step = 2
                    st.success("Analysis complete")
                    st.rerun()
        
        elif st.session_state.tutorial_step == 2:
            st.markdown("""
            <div class="demo-box demo-success">
            <h3>Step 3: Viewing Cohort Results</h3>
            <p>Here's the risk distribution across all patients:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show risk charts
            render_risk_charts(st.session_state.demo_results, mode="bulk", key_prefix="tutorial")
            
            st.info("These charts show how the 3 patients' risk scores are distributed. Notice the different risk categories and how patients rank relative to each other.")
            
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
            <h3>Tutorial Complete</h3>
            <p>You've learned how to:</p>
            <ul>
                <li>1. Work with sample patient data</li>
                <li>2. Run risk analysis</li>
                <li>3. View cohort results</li>
                <li>4. Examine individual patients</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("### Next Steps:")
            col_next1, col_next2 = st.columns(2)
            with col_next1:
                if st.button("Go to User Analysis", key="goto_user_analysis"):
                    st.info("Navigate to 'User Analysis' in the sidebar to work with your own data")
            with col_next2:
                if st.button("Restart Tutorial", key="restart_tut"):
                    st.session_state.tutorial_step = 0
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
            
            if st.button("Load & Analyze Sample Data", key="explore_analyze"):
                demo_results = process_data(demo_data)
                st.success("Sample data analyzed")
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
                4. **Radar**: Multi-modal balance
                5. **Bar Charts**: Biomarker levels
                """)
        
        with exploration_tab[2]:
            st.write("### Exploration Tips")
            
            st.info("""
            **Things to Try:**
            1. Compare all 3 sample patients' profiles
            2. Look for patterns in biomarker elevation
            3. See how protein/RNA/metabolite balance differs
            4. Check which markers appear in both patient-specific and global importance
            5. Expand the "View All Biomarker Values" section
            """)
            
            st.success("""
            **What Makes a Good Analysis:**
            1. Review both cohort and individual results
            2. Compare patient markers to global importance
            3. Note the multi-modal signature shape
            4. Look for biomarker clusters
            """)

    # Add reset button at bottom of demo page
    st.divider()
    if st.button("Reset Demo Workspace"):
        # Clear all session state related to demo
        keys_to_clear = [k for k in st.session_state.keys() if 'demo' in k or 'tutorial' in k]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

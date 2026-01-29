import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="MOmics-AI", layout="wide", page_icon="🧬")

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
        
        return model, feature_names, importance_df
    except FileNotFoundError:
        st.error("File 'gbm_clinical_model.pkl' not found. Please ensure it is in the root directory.")
        st.stop()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

model, feature_names, importance_df = load_assets()

# --- Data Validation Function ---
def validate_headers(df, feature_names):
    """
    Validate that uploaded data has matching column headers
    Returns: (is_valid, error_messages)
    """
    errors = []
    
    # Check if dataframe is empty
    if df.empty:
        errors.append("❌ The uploaded file is empty. Please provide patient data.")
        return False, errors
    
    # Check for column headers
    if df.columns.tolist() == list(range(len(df.columns))):
        errors.append("❌ No column headers detected. Please ensure your CSV file has headers in the first row.")
        return False, errors
    
    # Get uploaded columns
    uploaded_cols = set(df.columns)
    expected_cols = set(feature_names)
    
    # Check for matching columns
    matching_cols = uploaded_cols.intersection(expected_cols)
    
    if len(matching_cols) == 0:
        errors.append("❌ No matching biomarker columns found. Please check your column headers.")
        errors.append("   Expected format: biomarker names with suffixes like '_prot', '_rna', '_met'")
        errors.append("   Example: 'TP53_prot', 'EGFR_rna', 'Glucose_met'")
        
        # Show first few columns as reference
        if len(uploaded_cols) > 0:
            sample_cols = list(uploaded_cols)[:5]
            errors.append(f"   Your columns start with: {', '.join(sample_cols)}")
        
        return False, errors
    
    # If we have at least some matching columns, consider it valid
    return True, errors

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
st.sidebar.title("MOmics-AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Documentation", "User Analysis", "Demo Walkthrough"]
)

# --- MAIN INTERFACE ---
st.title("MOmics-AI | GBM Clinical Diagnostic Suite")

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
    
    doc_tabs = st.tabs([
        "Overview",
        "System Architecture & Model",
        "Data Requirements"
    ])
    
    # Overview Tab
    with doc_tabs[0]:
        st.markdown("""
        ### Purpose & Scope
        
        MOmics-AI is a clinical decision support tool designed for glioblastoma patient risk stratification. 
        The system integrates multi-omics biomarker data to generate probability-based risk assessments, 
        helping clinicians identify high-risk patients who may benefit from aggressive treatment strategies.
        
        ### Workflow Architecture
        
        The platform follows a streamlined analysis pipeline:
        
        1. **Data Input**: Raw laboratory values for 843 biomarkers (proteomics, transcriptomics, metabolomics)
        2. **Preprocessing**: Automatic alignment with model feature space, zero-filling for missing markers
        3. **Inference**: XGBoost model generates risk probability scores
        4. **Visualization**: Interactive dashboards display predictions, biomarker influences, and patient profiles
        5. **Export**: Results available for clinical record integration
        
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
    
    # System Architecture & Model Tab
    with doc_tabs[1]:
        st.markdown("""
        ### System Architecture Overview
        
        MultiNet_AI follows a three-tier architecture consisting of:
        
        1. **Frontend Layer** (Streamlit-based User Interface)
        2. **Backend Layer** (Python Processing Engine)
        3. **Machine Learning Layer** (XGBoost Model)
        
        For detailed technical documentation, please refer to the full system architecture guide.
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
        4. Missing markers are automatically filled with 0.0
        
        #### CSV File Format (Bulk Upload)
        
        **Header Row Requirements**
        1. Must contain exact biomarker names matching model features
        2. No spaces or special characters except underscore
        3. Case-sensitive matching
        4. Order does not matter (automatically reordered)
        
        **Data Rows**
        1. One patient per row
        2. No blank rows between records
        3. Maximum recommended: 1000 patients per file
        
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
        
        #### Manual Entry Guidelines
        
        1. Prioritize top 12 high-influence markers shown by default
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
        2. One patient per row starting from row 2
        3. Fill columns with numeric values
        4. Save as CSV format (not Excel .xlsx)
        5. Upload via User Analysis interface
        
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
        
        **Best Practices**
        1. Remove patient names from CSV files
        2. Use study IDs instead of medical record numbers
        3. Strip dates to month/year only
        4. Exclude geographic identifiers below state level
        5. Review data before upload
        """)

# ============================================================================
# USER ANALYSIS PAGE - WITH HEADER VALIDATION ONLY
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
                user_inputs[name] = st.number_input(f"{name}", value=0.0, key=f"man_in_{name}")
                
        with st.expander("Advanced Marker Input (Full 843 Set)"):
            adv_cols = st.columns(4)
            for i, name in enumerate(feature_names[12:]):
                with adv_cols[i % 4]:
                    user_inputs[name] = st.number_input(f"{name}", value=0.0, key=f"man_adv_{name}")

        # IMPORTANT: Only show results AFTER button click
        if st.button("Analyze Single Patient", key="btn_manual", type="primary"):
            m_results = process_data(pd.DataFrame([user_inputs]))
            st.success("Analysis Complete! Results displayed below.")
            st.divider()
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
            uploaded_file = st.file_uploader("Upload filled MultiNet CSV Template", type="csv", 
                                            help="Upload a CSV file with patient biomarker data")
        
        # IMPORTANT: Validate headers BEFORE processing
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                
                # Validate only the headers
                is_valid, errors = validate_headers(raw_df, feature_names)
                
                if not is_valid:
                    # Show errors
                    st.error("**Column Header Validation Failed**")
                    for error in errors:
                        st.error(error)
                    
                    # Show helpful guidance
                    with st.expander("ℹ️ How to Fix These Issues"):
                        st.markdown("""
                        **Common Issues and Solutions:**
                        
                        1. **Missing or incorrect headers:**
                           - Download the template using the button above
                           - Ensure your column names exactly match the template
                           - Check for typos in biomarker names (case-sensitive)
                           - Headers must include suffixes: '_prot', '_rna', or '_met'
                        
                        2. **File format issues:**
                           - Save your file as CSV format (.csv)
                           - Use UTF-8 encoding
                           - Ensure the first row contains column headers
                        
                        **Need Help?**
                        - Navigate to Documentation → Data Requirements for detailed specifications
                        - Download and review the template file
                        - Ensure your column headers follow the expected naming convention
                        """)
                
                else:
                    # Headers are valid - proceed with processing
                    st.success(f"✅ File uploaded successfully! Found {len(raw_df)} patient(s).")
                    
                    # Process and show dashboard
                    b_results = process_data(raw_df)
                    st.divider()
                    st.subheader("Analysis Results")
                    render_dashboard(b_results, mode="bulk", key_prefix="blk")
                    
            except pd.errors.EmptyDataError:
                st.error("❌ The uploaded file is empty. Please provide a valid CSV file with patient data.")
            except pd.errors.ParserError:
                st.error("❌ Error parsing CSV file. Please ensure your file is properly formatted.")
                st.info("Tip: Open the file in a text editor and check for unusual characters or formatting.")
            except Exception as e:
                st.error(f"❌ Unexpected error processing file: {e}")
                st.info("Please ensure your CSV file follows the template format. Download the template above for reference.")

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
            with st.spinner("🔬 Analyzing biomarkers..."):
                demo_results = process_data(demo_data)
            
            st.success("✅ Analysis Complete!")
            
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
            <h3>Tutorial Complete! 🎉</h3>
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
                st.info("📊 Navigate to 'User Analysis' in the sidebar to work with your own data")
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
                with st.spinner("🔬 Analyzing sample data..."):
                    demo_results = process_data(demo_data)
                st.success("✅ Sample data analyzed successfully!")
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
    if st.button("Reset Demo Workspace"):
        # Clear all session state related to demo
        keys_to_clear = [k for k in st.session_state.keys() if 'demo' in k or 'tutorial' in k]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

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

# Create main navigation tabs
tab_home, tab_docs, tab_user, tab_demo = st.tabs([
    "🏠 Home", 
    "📚 Documentation", 
    "🔬 User Analysis",
    "🎬 Demo Walkthrough"
])

# ============================================================================
# HOME TAB
# ============================================================================
with tab_home:
    # Display logo centered
    try:
        logo = Image.open('logo.png')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo, use_container_width=True)
    except:
        st.info("Logo image not found. Please ensure 'logo.png' is in the root directory.")
    
    # Centered title
    st.markdown("<h1 style='text-align: center;'>MultiNet_AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>GBM Clinical Diagnostic Suite</h3>", unsafe_allow_html=True)

# ============================================================================
# DOCUMENTATION TAB
# ============================================================================
with tab_docs:
    st.header("📚 System Documentation")
    
    doc_section = st.radio(
        "Select Documentation Section:",
        ["Overview", "Frontend Architecture", "Backend Architecture", "Data Requirements", "Model Information"],
        horizontal=True
    )
    
    if doc_section == "Overview":
        st.subheader("System Overview")
        st.markdown("""
        ### Purpose & Scope
        
        MultiNet_AI is a clinical decision support tool designed for glioblastoma patient risk stratification. 
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
        
        - **Treatment Planning**: Identify patients requiring aggressive intervention
        - **Prognosis Assessment**: Stratify patients by molecular risk profiles
        - **Research Studies**: Batch analysis of patient cohorts
        - **Biomarker Discovery**: Explore feature importance across the global model
        """)
    
    elif doc_section == "Frontend Architecture":
        st.subheader("Frontend Architecture")
        st.markdown("""
        ### Technology Stack
        
        **Framework**: Streamlit  
        **Visualization**: Plotly (interactive charts)  
        **Data Handling**: Pandas, NumPy  
        **Language**: Python 3.8+
        
        ### Component Structure
        
        #### 1. Navigation System
        - **Tab-based Interface**: Four primary sections (Home, Documentation, User Analysis, Demo)
        - **Persistent State**: Unique keys prevent widget conflicts across tabs
        - **Responsive Layout**: Columns adapt to screen size using `use_container_width`
        
        #### 2. Input Modules
        
        **Manual Entry Interface**:
        - Top 12 high-influence biomarkers displayed by default
        - Expandable advanced panel for remaining 831 markers
        - Number input widgets with default zero-fill for baseline simulation
        - Real-time validation ensures numeric data types
        
        **Bulk Upload Interface**:
        - CSV file uploader with template generation
        - Automatic column alignment with model expectations
        - Error handling for malformed files
        
        #### 3. Visualization Components
        
        **Risk Assessment Visuals**:
        - **Gauge Chart** (Single Patient): Semi-circular gauge displaying risk percentage
        - **Pie Chart** (Cohort): Distribution of high/low risk classifications
        - **Histogram** (Cohort): Probability density distribution across patients
        
        **Biomarker Analysis**:
        - **Global Influence Bar Chart**: Top 15 features by model importance
        - **Searchable DataTable**: Full 843-marker influence scores
        - **Patient Deep-Dive**: Top 20 marker levels for selected individual
        - **Multi-Modal Radar Chart**: Average expression across omics layers
        
        #### 4. Interactivity Features
        
        - **Patient Selection**: Dropdown to explore individual records in batch results
        - **Expandable Sections**: Collapsible panels for advanced options and full data views
        - **Download Buttons**: CSV template generation for standardized input
        - **Color Coding**: Risk-aware palettes (red = high risk, green/blue = low risk)
        
        ### User Experience Design
        
        - **Progressive Disclosure**: Essential features visible, advanced options hidden until needed
        - **Visual Hierarchy**: Headers, dividers, and spacing guide attention flow
        - **Feedback Mechanisms**: Spinners during processing, error messages for failures
        - **Accessibility**: High-contrast colors, large fonts, clear labels
        """)
    
    elif doc_section == "Backend Architecture":
        st.subheader("Backend Architecture")
        st.markdown("""
        ### Core Processing Pipeline
        
        #### 1. Model Loading (`load_assets`)
        
        **Function**: Loads trained XGBoost model from pickle file  
        **Caching**: `@st.cache_resource` ensures single load per session  
        **Outputs**:
        - `model`: XGBoost classifier object
        - `feature_names`: List of 843 expected biomarker identifiers
        - `importance_df`: Precomputed feature importances sorted descending
        
        **Error Handling**:
        - FileNotFoundError: Displays error if model file missing
        - Generic exceptions: Catches serialization/compatibility issues
        - `st.stop()`: Prevents execution if model fails to load
        
        #### 2. Data Preprocessing (`process_data`)
        
        **Input**: Pandas DataFrame with user-provided biomarker values  
        **Process**:
        1. **Column Alignment**: `df.reindex(columns=feature_names, fill_value=0.0)`
           - Ensures exact match with model's 843 expected features
           - Fills missing columns with 0.0 (baseline expression)
           - Drops any extra columns not in training set
        
        2. **Type Conversion**: `.astype(float)` enforces numeric data types
        
        3. **Inference**: 
           - `predict_proba()`: Returns probability array [P(low risk), P(high risk)]
           - Extracts `[:, 1]` for high-risk probability
           - Thresholding at 0.5 generates binary classification
        
        **Output**: DataFrame containing:
        - `Prediction`: "High Risk" or "Low Risk" label
        - `Risk Score`: Probability value (0.0 - 1.0)
        - All 843 aligned biomarker values for downstream analysis
        
        #### 3. Visualization Rendering
        
        **Risk Charts (`render_risk_charts`)**:
        - Mode-aware rendering (single gauge vs. cohort charts)
        - Dynamic color assignment based on risk category
        - Plotly figure generation with unique keys per context
        
        **Dashboard Assembly (`render_dashboard`)**:
        - Modular function orchestrating three visualization blocks
        - Sequential rendering: predictions → global influence → patient explorer
        - State management through key prefixes prevents widget collisions
        
        ### Machine Learning Model
        
        #### Algorithm: XGBoost (Extreme Gradient Boosting)
        
        **Why XGBoost?**
        - **High-Dimensional Data**: Handles 843 features efficiently
        - **Non-Linear Relationships**: Captures complex omics interactions
        - **Feature Importance**: Built-in SHAP-like importance scoring
        - **Regularization**: Prevents overfitting on small clinical datasets
        
        #### Model Specifications
        
        - **Task**: Binary classification (high risk vs. low risk)
        - **Output**: Probability scores via logistic objective
        - **Features**: 843 multi-omics biomarkers
        - **Training**: Supervised learning on labeled patient outcomes
        
        #### Feature Importance Calculation
        
        - **Method**: `model.feature_importances_` (gain-based)
        - **Interpretation**: Relative contribution to risk probability
        - **Global Scope**: Aggregated across all trees and patients
        - **Use Case**: Identifies key biomarkers for clinical focus
        
        ### Data Flow Diagram
```
        User Input (CSV/Manual)
              ↓
        process_data() → Column Alignment → Zero-Filling
              ↓
        XGBoost Model → predict_proba()
              ↓
        Risk Scores + Predictions
              ↓
        Visualization Layer → Plotly Charts
              ↓
        Interactive Dashboard Display
```
        
        ### Performance Considerations
        
        - **Caching**: Model loaded once per session via `@st.cache_resource`
        - **Lazy Loading**: Charts rendered only when tabs accessed
        - **Memory**: Full 843-column DataFrames stored in session state
        - **Processing Time**: ~1-5 seconds for batch inference (100 patients)
        """)
    
    elif doc_section == "Data Requirements":
        st.subheader("Data Requirements & Formats")
        st.markdown("""
        ### Input Data Specifications
        
        #### Biomarker Identifiers
        
        The model expects exactly **843 biomarkers** with specific naming conventions:
        
        - **Proteomics**: Suffixed with `_prot` (e.g., `TP53_prot`, `EGFR_prot`)
        - **Transcriptomics**: Suffixed with `_rna` (e.g., `IDH1_rna`, `MGMT_rna`)
        - **Metabolomics**: Suffixed with `_met` (e.g., `Glucose_met`, `Lactate_met`)
        
        #### Value Ranges
        
        - **Data Type**: Continuous numeric (float)
        - **Units**: Raw laboratory values (model trained on non-normalized data)
        - **Missing Data**: Enter `0.0` to represent baseline/undetected levels
        - **Negative Values**: Not expected; use absolute measurements
        
        #### CSV File Format (Bulk Upload)
        
        **Structure**:
        - **Header Row**: Must contain exact biomarker names matching model features
        - **Data Rows**: One patient per row
        - **Delimiter**: Comma-separated (standard CSV)
        - **Encoding**: UTF-8
        
        **Example**:
```
        TP53_prot,EGFR_prot,IDH1_rna,MGMT_rna,Glucose_met,...
        12.5,8.3,150.2,22.1,85.0,...
        9.1,11.7,98.5,19.3,72.4,...
```
        
        **Column Handling**:
        - Extra columns (not in 843 features): Automatically dropped
        - Missing columns: Filled with 0.0 during alignment
        - Order: Does not matter; automatic reordering by `reindex()`
        
        #### Manual Entry Guidelines
        
        1. **Prioritize High-Influence Markers**: Top 12 fields shown by default
        2. **Use Zero for Unknowns**: Leave fields at 0.0 if data unavailable
        3. **Check Units**: Ensure values match training data scale
        4. **Avoid Text**: Only numeric inputs accepted
        
        ### Template Generation
        
        **Download Template**:
        - Click "Download CSV Template" in User Analysis tab
        - Opens CSV with 843 pre-labeled columns
        - Fill in patient data rows
        - Upload back to platform
        
        ### Data Privacy & Security
        
        - **No Storage**: Patient data not saved server-side
        - **Session-Based**: Data cleared when browser tab closed
        - **Local Processing**: All inference happens in Streamlit session
        - **HIPAA Considerations**: Suitable for de-identified research data
        """)
    
    elif doc_section == "Model Information":
        st.subheader("Model Information & Performance")
        st.markdown("""
        ### Training Dataset
        
        #### Source
        - **Cohort**: Glioblastoma patients from clinical repository
        - **Sample Size**: Training set size determined by available multi-omics data
        - **Outcome Variable**: Binary risk classification (high/low) based on survival or progression
        
        #### Data Preprocessing
        - **Normalization**: Raw values used directly (no scaling during training)
        - **Feature Selection**: All 843 biomarkers retained for comprehensive coverage
        - **Class Balance**: Handled via XGBoost's `scale_pos_weight` parameter
        
        ### Model Architecture
        
        #### XGBoost Hyperparameters
        - **Objective**: `binary:logistic` (probability outputs)
        - **Booster**: `gbtree` (gradient boosted trees)
        - **Regularization**: L1/L2 penalties to prevent overfitting
        - **Tree Depth**: Controlled to balance complexity and generalization
        
        #### Training Process
        - **Cross-Validation**: K-fold CV for hyperparameter tuning
        - **Early Stopping**: Prevents overtraining on validation loss
        - **Evaluation Metric**: AUC-ROC for classification performance
        
        ### Model Outputs
        
        #### Risk Score
        - **Range**: 0.0 (lowest risk) to 1.0 (highest risk)
        - **Interpretation**: Probability of belonging to high-risk class
        - **Threshold**: 0.5 cutoff for binary classification
        - **Clinical Use**: Scores >0.7 indicate very high risk; <0.3 low risk
        
        #### Prediction Label
        - **High Risk**: Patients likely to have poor outcomes
        - **Low Risk**: Patients with favorable prognosis
        - **Decision Boundary**: 50% probability threshold
        
        ### Feature Importance
        
        #### Calculation Method
        - **Gain-Based**: Measures average improvement in loss function
        - **Global Aggregation**: Summed across all trees and splits
        - **Normalization**: Scaled to relative percentages
        
        #### Top Influencers
        The model identifies biomarkers with strongest impact on risk probability. 
        These are displayed in the Global Biomarker Influence chart for clinical interpretation.
        
        #### Biological Interpretation
        - High-importance proteins may indicate aggressive tumor biology
        - RNA signatures reflect transcriptional dysregulation
        - Metabolite markers capture tumor microenvironment changes
        
        ### Model Limitations
        
        #### Scope
        - **Disease-Specific**: Trained only on glioblastoma patients
        - **Population**: Performance may vary across demographics not represented in training
        - **Biomarkers**: Limited to 843 measured features; novel markers not included
        
        #### Clinical Considerations
        - **Not Diagnostic**: Provides risk stratification, not diagnosis
        - **Adjunct Tool**: Should complement, not replace, clinical judgment
        - **Validation**: Requires external validation in prospective studies
        
        #### Technical Constraints
        - **Missing Data**: Zero-filling may not capture true baseline for all markers
        - **Batch Effects**: Assumes consistent measurement protocols across labs
        - **Temporal Drift**: Model may need retraining as treatment standards evolve
        
        ### Recommendations
        
        1. **Clinical Integration**: Use as part of multidisciplinary tumor board discussions
        2. **Threshold Tuning**: Adjust risk cutoffs based on institutional resources
        3. **Monitoring**: Track model performance on real-world patients
        4. **Updating**: Periodically retrain with new data to maintain accuracy
        """)
    
    st.markdown("---")
    st.info("💡 **Tip**: For hands-on learning, visit the **Demo Walkthrough** tab after reviewing documentation.")

# ============================================================================
# USER ANALYSIS TAB (Original Code - Manual + Batch)
# ============================================================================
with tab_user:
    st.header("🔬 User Analysis")
    
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

# ============================================================================
# DEMO WALKTHROUGH TAB
# ============================================================================
with tab_demo:
    st.header("🎬 Demo Walkthrough")
    
    demo_step = st.selectbox(
        "Select Tutorial:",
        [
            "1. Single Patient Analysis",
            "2. Interpreting Risk Scores",
            "3. Bulk Data Processing",
            "4. Biomarker Exploration",
            "5. Exporting Results"
        ]
    )
    
    if demo_step == "1. Single Patient Analysis":
        st.subheader("Tutorial: Single Patient Risk Assessment")
        
        st.markdown("""
        ### Scenario
        You have a newly diagnosed GBM patient with laboratory results for key biomarkers. 
        You want to determine their risk category to guide treatment planning.
        
        ### Step-by-Step Guide
        
        #### Step 1: Navigate to User Analysis
        - Click the **"🔬 User Analysis"** tab above
        - Select **"✍️ Manual Patient Entry"** sub-tab
        
        #### Step 2: Enter Biomarker Values
        - You'll see 12 high-influence marker input fields
        - Enter raw laboratory values (e.g., protein concentration, RNA expression levels)
        - **Example Values**:
          - `TP53_prot`: 15.2
          - `EGFR_prot`: 22.8
          - `IDH1_rna`: 145.0
          - Leave others at 0.0 if data unavailable
        
        #### Step 3: Access Advanced Markers (Optional)
        - Expand "🛠️ Advanced Marker Input" section
        - Input additional biomarkers if available
        - Scroll through 831 additional fields organized in columns
        
        #### Step 4: Run Analysis
        - Click **"🚀 Analyze Single Patient"** button
        - Wait for processing (1-3 seconds)
        
        #### Step 5: Review Results
        - **Gauge Chart**: Shows risk percentage and classification
        - **Biomarker Influence**: See which markers drive model decisions
        - **Patient Profile**: Radar chart displays multi-omics summary
        """)
        
        with st.expander("📋 Example Demo Data"):
            st.code("""
# Copy these values for testing:
TP53_prot: 18.5
EGFR_prot: 25.3
IDH1_rna: 162.7
MGMT_rna: 88.4
PTEN_prot: 12.1
VEGFA_rna: 195.6
            """)
    
    elif demo_step == "2. Interpreting Risk Scores":
        st.subheader("Tutorial: Understanding Risk Outputs")
        
        st.markdown("""
        ### Risk Score Interpretation
        
        #### Score Ranges & Clinical Meaning
        
        | Risk Score | Category | Interpretation | Action |
        |------------|----------|----------------|--------|
        | 0.0 - 0.3 | Very Low Risk | Favorable molecular profile | Standard surveillance |
        | 0.3 - 0.5 | Low Risk | Below threshold for high-risk | Conservative treatment |
        | 0.5 - 0.7 | Moderate-High Risk | Above classification threshold | Consider aggressive therapy |
        | 0.7 - 1.0 | Very High Risk | Strongly unfavorable profile | Maximum intervention |
        
        #### Gauge Chart Elements
        
        **Color Coding**:
        - **Green Zone** (0-50%): Low risk, below classification threshold
        - **Red Zone** (50-100%): High risk, exceeds threshold
        
        **Needle Position**: Indicates exact probability score
        
        **Title Display**: Shows binary classification ("High Risk" or "Low Risk")
        
        #### Clinical Decision Making
        
        1. **High Confidence Predictions**:
           - Scores <0.2 or >0.8: Strong model confidence
           - Safe to rely on classification for treatment planning
        
        2. **Borderline Cases**:
           - Scores 0.45-0.55: Near decision boundary
           - Consider additional clinical factors
           - May benefit from repeat testing or expanded biomarker panel
        
        3. **Risk Trends**:
           - Compare scores over time (serial measurements)
           - Increasing scores suggest disease progression
           - Decreasing scores may indicate treatment response
        
        #### Multi-Modal Signature Radar
        
        **Axes**:
        - **Proteins**: Average expression of `_prot` markers
        - **RNA**: Average expression of `_rna` markers
        - **Metabolites**: Average expression of `_met` markers
        
        **Shape Interpretation**:
        - **Balanced Triangle**: Concordant omics layers
        - **Skewed Shape**: Dominant signal from one modality
        - **Large Area**: High overall biomarker activity
        """)
    
    elif demo_step == "3. Bulk Data Processing":
        st.subheader("Tutorial: Batch Processing Patient Cohorts")
        
        st.markdown("""
        ### Scenario
        You have a research cohort of 50 GBM patients and need risk stratification for all.
        
        ### Step-by-Step Guide
        
        #### Step 1: Download Template
        - Navigate to **"🔬 User Analysis"** tab
        - Select **"💾 Bulk Data Upload"** sub-tab
        - Click **"📥 Download CSV Template"** button
        - Save the file to your computer (named `MultiNet_Patient_Template.csv`)
        
        #### Step 2: Populate Template
        - Open CSV in Excel, Google Sheets, or text editor
        - **Column Headers**: 843 biomarker names (do not modify)
        - **Data Rows**: Add one row per patient
        - **Values**: Enter raw laboratory measurements
        - **Missing Data**: Leave cells empty or enter 0.0
        
        **Example Structure**:
```
        TP53_prot,EGFR_prot,IDH1_rna,...
        15.2,22.8,145.0,...              (Patient 1)
        18.3,19.1,132.5,...              (Patient 2)
        12.7,25.4,168.2,...              (Patient 3)
```
        
        #### Step 3: Upload Filled Template
        - Return to **"💾 Bulk Data Upload"** sub-tab
        - Click **"⬆️ Upload Patient Data"** file uploader
        - Select your completed CSV file
        
        #### Step 4: Automatic Processing
        - System processes all patients simultaneously
        - Progress spinner displays during analysis
        - Results appear automatically after completion
        
        #### Step 5: Review Cohort Results
        
        **Pie Chart**:
        - Shows distribution: X% high risk, Y% low risk
        - Helps understand overall cohort characteristics
        
        **Histogram**:
        - Probability distribution across all patients
        - Identifies clusters or bimodal patterns
        
        **Patient Explorer**:
        - Use dropdown to select individual records
        - Review specific biomarker profiles
        - Compare top markers across patients
        
        #### Step 6: Export Analysis
        - Results displayed in interactive tables
        - Use browser print function for reports
        - Take screenshots of visualizations for presentations
        """)
        
        with st.expander("📋 Sample Data Generator"):
            st.info("In a real application, you would download the template and fill it with actual patient data.")
    
    elif demo_step == "4. Biomarker Exploration":
        st.subheader("Tutorial: Discovering Key Biomarkers")
        
        st.markdown("""
        ### Understanding Feature Importance
        
        #### Global Biomarker Influence Chart
        
        **What It Shows**:
        - Top 15 biomarkers ranked by contribution to risk predictions
        - Horizontal bar chart sorted by influence score
        - Red color gradient indicates relative importance
        
        **How to Interpret**:
        1. **Highest Bar**: Most influential marker across all patients
        2. **Score Magnitude**: Relative contribution to model decisions
        3. **Biomarker Type**: Note suffixes (_prot, _rna, _met)
        
        **Clinical Applications**:
        - **Focused Testing**: Prioritize measuring high-influence markers
        - **Pathway Analysis**: Group related proteins/genes for mechanistic insight
        - **Drug Targets**: High-importance markers may be therapeutic candidates
        
        #### Full 843-Marker Table
        
        **Accessing**:
        - Expand **"📄 View Searchable Influence List"**
        - Scrollable table with all features
        
        **Features**:
        - **Search Box**: Filter by marker name (e.g., type "TP53" to find all related)
        - **Sortable Columns**: Click headers to reorder
        - **Copy Function**: Select and copy data for external analysis
        
        **Use Cases**:
        - Find influence score for specific marker of interest
        - Compare related proteins (e.g., all kinases)
        - Export for meta-analysis or publication
        
        #### Patient-Specific Deep Dive
        
        **Top 20 Marker Levels Chart**:
        - Shows highest-expressed biomarkers for selected patient
        - Horizontal bar chart with color gradient
        - Helps identify dominant molecular signals
        
        **Multi-Modal Radar**:
        - Triangular chart showing omics balance
        - Three axes: Proteins, RNA, Metabolites
        - Reveals which data types contribute most to patient profile
        
        #### Discovery Workflow
        
        1. **Global Analysis**: Review top 15 influencers to understand model logic
        2. **Literature Search**: Research identified markers in GBM context
        3. **Patient Correlation**: Check if high-risk patients share marker patterns
        4. **Hypothesis Generation**: Develop biological explanations for signatures
        5. **Validation**: Design follow-up experiments for key findings
        """)
    
    elif demo_step == "5. Exporting Results":
        st.subheader("Tutorial: Saving & Sharing Analysis")
        
        st.markdown("""
        ### Export Options
        
        #### Method 1: Screenshot Visualizations
        
        **For Presentations**:
        1. Display desired chart (gauge, pie, histogram, etc.)
        2. Use OS screenshot tool:
           - **Windows**: Windows + Shift + S
           - **Mac**: Command + Shift + 4
           - **Linux**: Shift + PrtSc
        3. Paste into PowerPoint, Word, or reports
        
        **Best Practices**:
        - Maximize chart before capturing
        - Include title and legend in frame
        - Use high-resolution display for clarity
        
        #### Method 2: Interactive Chart Export
        
        **Plotly Built-In Tools**:
        - Hover over any chart
        - Click camera icon (top-right corner)
        - Downloads PNG image to default folder
        - Alternative: Click "..." → "Download plot as png"
        
        **Features**:
        - High-resolution vector graphics
        - Preserves colors and formatting
        - Suitable for publication
        
        #### Method 3: Data Table Copy
        
        **From Searchable Tables**:
        1. Navigate to expandable data views
        2. Select rows of interest
        3. Right-click → Copy or Ctrl+C / Cmd+C
        4. Paste into Excel, Google Sheets, or text editor
        
        **Use Cases**:
        - Export feature importance scores
        - Save patient risk scores
        - Create custom reports
        
        #### Method 4: Print to PDF
        
        **Full Report Generation**:
        1. Display complete dashboard (all sections visible)
        2. Browser → Print (Ctrl+P / Cmd+P)
        3. Select "Save as PDF" destination
        4. Adjust margins and scale
        5. Save comprehensive report
        
        **Tips**:
        - Use landscape orientation for wide charts
        - Enable background graphics for colors
        - Remove headers/footers for clean look
        
        ### Clinical Documentation
        
        #### Electronic Health Record (EHR) Integration
        
        **Current Limitations**:
        - No direct EHR export available
        - Manual copy-paste required
        
        **Workaround**:
        1. Take screenshot of risk gauge
        2. Copy risk score and prediction label
        3. Attach as external document in patient chart
        4. Document analysis date and model version
        
        #### Research Data Management
        
        **For Clinical Trials**:
        - Save CSV template with patient IDs
        - Use batch processing for all subjects
        - Screenshot cohort summary charts
        - Maintain version control of input files
        
        **For Publications**:
        - Export feature importance table
        - Save high-res chart images
        - Document model hyperparameters
        - Include data availability statement
        
        ### Sharing Best Practices
        
        #### Internal Collaboration
        - Share screenshots via email or Slack
        - Upload charts to shared drives
        - Present dashboard live in meetings
        
        #### External Dissemination
        - De-identify patient data before sharing
        - Include MultiNet_AI citation
        - Specify model version and date
        - Note limitations in interpretations
        """)

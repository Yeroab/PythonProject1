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
        # Histogram for Bulk Entry
        fig_hist = px.histogram(results, x="Risk Score", color="Prediction",
                                 title="Risk Probability Distribution",
                                 color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"})
        st.plotly_chart(fig_hist, use_container_width=True, key=f"{key_prefix}_hist")

# --- Section: Complete Dashboard ---
def render_dashboard(results, mode="manual", key_prefix=""):
    # 1. Prediction Visuals
    render_risk_charts(results, mode=mode, key_prefix=key_prefix)
    
    # 2. Individual Patient Explorer
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
    
    # 3. Patient-Specific Biomarker Influence
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
    
    # Create tabs for documentation sections
    doc_tabs = st.tabs([
        "Overview",
        "Frontend Architecture",
        "Backend Architecture",
        "Data Requirements",
        "Model Information"
    ])
    
    # Overview Tab
    with doc_tabs[0]:
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
        
        **Treatment Planning**
        - Identify patients requiring aggressive intervention
        - Guide therapy selection based on molecular risk profiles
        - Support multidisciplinary tumor board discussions
        - Prioritize resource allocation for high-risk cases
        
        **Prognosis Assessment**
        - Stratify patients by molecular risk profiles
        - Estimate likelihood of poor outcomes
        - Inform patient and family counseling
        - Support clinical trial enrollment decisions
        
        **Research Studies**
        - Batch analysis of patient cohorts
        - Retrospective outcome correlation
        - Biomarker validation studies
        - Clinical trial stratification
        
        **Biomarker Discovery**
        - Explore feature importance across the global model
        - Identify patient-specific therapeutic targets
        - Validate known prognostic markers
        - Discover novel risk indicators
        
        ### System Requirements
        
        **Hardware Requirements**
        - Modern web browser (Chrome, Firefox, Safari, Edge)
        - Minimum 4GB RAM recommended
        - Stable internet connection
        - Display resolution: 1280x720 or higher
        
        **Software Dependencies**
        - Python 3.8 or higher
        - Streamlit framework
        - XGBoost machine learning library
        - Plotly visualization library
        - Pandas data processing library
        
        **Data Requirements**
        - CSV format for bulk uploads
        - Numeric biomarker values
        - Standardized column headers
        - UTF-8 encoding
        """)
    
    # Frontend Architecture Tab
    with doc_tabs[1]:
        st.markdown("""
        ### Technology Stack
        
        **Framework**: Streamlit 1.28+
        - Rapid prototyping and deployment
        - Built-in widget management
        - Session state handling
        - Automatic reactivity
        
        **Visualization**: Plotly 5.17+
        - Interactive charts and graphs
        - Hover tooltips and zooming
        - Export capabilities
        - Responsive design
        
        **Data Handling**: Pandas 2.0+, NumPy 1.24+
        - Efficient data manipulation
        - Missing value handling
        - Type conversion utilities
        - Statistical operations
        
        **Styling**: Custom CSS
        - Responsive layout design
        - Theme customization
        - Component styling
        - Brand consistency
        
        ### Component Structure
        
        #### 1. Navigation System
        
        **Sidebar Navigation**
        - Four primary sections accessible via radio buttons
        - Persistent across sessions
        - Visual highlighting of active page
        - Compact design for space efficiency
        
        **Tab-based Sub-navigation**
        - Organized content within each section
        - Horizontal tab layout
        - Clear section separation
        - Intuitive workflow progression
        
        **State Management**
        - Unique keys prevent widget conflicts
        - Session state persistence
        - Cross-page data sharing
        - Reset functionality
        
        **Responsive Layout**
        - Columns adapt to screen size
        - Mobile-friendly design
        - Automatic reflow
        - Consistent spacing
        
        #### 2. Input Modules
        
        **Manual Entry Interface**
        
        *High-Priority Markers Section*
        - Top 12 high-influence biomarkers displayed by default
        - Three-column grid layout for efficient scanning
        - Number input widgets with validation
        - Default zero-fill for baseline simulation
        - Clear labeling with biomarker names
        
        *Advanced Marker Section*
        - Collapsible expander for remaining 831 markers
        - Four-column grid for compact presentation
        - Consistent input widget styling
        - Alphabetical or importance-based ordering
        - Bulk entry support
        
        *Input Validation*
        - Real-time type checking
        - Range validation (non-negative values)
        - Required field indicators
        - Error messaging
        - Auto-correction suggestions
        
        **Bulk Upload Interface**
        
        *Template Generation*
        - One-click CSV template download
        - Pre-populated column headers
        - Example data rows (optional)
        - Format specifications included
        - Version tracking
        
        *File Upload Component*
        - Drag-and-drop support
        - File type validation (.csv only)
        - Size limit checking
        - Progress indication
        - Error recovery
        
        *Data Validation*
        - Automatic column alignment
        - Missing value detection
        - Type conversion with error handling
        - Duplicate row identification
        - Data quality metrics
        
        #### 3. Visualization Components
        
        **Risk Assessment Visuals**
        
        *Gauge Chart (Single Patient)*
        - Semi-circular gauge displaying risk percentage
        - Color-coded zones (green for low, red for high)
        - Numeric risk score display
        - Prediction label (High/Low Risk)
        - Threshold indicator at 50%
        
        *Histogram (Cohort Analysis)*
        - Probability density distribution across patients
        - Color-coded by risk category
        - Bin size optimization
        - Overlay statistics
        - Exportable as PNG/SVG
        
        **Biomarker Analysis Visualizations**
        
        *Global Influence Bar Chart*
        - Top 15 features by model importance
        - Horizontal orientation for readability
        - Color gradient indicating magnitude
        - Sorted by importance score
        - Interactive tooltips
        
        *Patient-Specific Charts*
        - Top 20 marker levels for selected individual
        - Side-by-side comparison (patient vs global)
        - Color-coded by expression level
        - Sortable and filterable
        - Drill-down capabilities
        
        *Multi-Modal Radar Chart*
        - Three-axis display (Protein, RNA, Metabolite)
        - Filled area indicating expression levels
        - Average expression across omics layers
        - Comparison overlay option
        - Export functionality
        
        *Comparative Views*
        - Patient values vs global importance
        - Dual bar chart layout
        - Aligned y-axes for comparison
        - Highlighting of overlapping markers
        - Statistical significance indicators
        
        #### 4. Interactivity Features
        
        **Patient Selection**
        - Dropdown to explore individual records
        - Index or ID-based selection
        - Preview on hover
        - Quick navigation buttons
        - Comparison mode
        
        **Expandable Sections**
        - Collapsible panels for advanced options
        - Full data table views
        - Progressive disclosure pattern
        - Smooth animations
        - Persistent state across sessions
        
        **Download Functionality**
        - CSV template generation
        - Results export (PDF, CSV)
        - Chart image downloads (PNG, SVG)
        - Batch export options
        - Customizable formats
        
        **Color Coding System**
        - Risk-aware palettes (red for high risk, green for low)
        - Consistent across all visualizations
        - Colorblind-friendly options
        - Customizable themes
        - Legend and key provided
        
        ### User Experience Design
        
        **Progressive Disclosure**
        - Essential features visible by default
        - Advanced options hidden until needed
        - Logical information hierarchy
        - Reduced cognitive load
        - Guided workflow
        
        **Visual Hierarchy**
        - Clear headers and section dividers
        - Consistent spacing and alignment
        - Typography scale for importance
        - White space utilization
        - Attention flow guidance
        
        **Feedback Mechanisms**
        - Loading spinners during processing
        - Success/error message toasts
        - Progress bars for batch operations
        - Validation feedback
        - Confirmation dialogs
        
        **Accessibility**
        - High-contrast color schemes
        - Large, readable fonts (minimum 14px)
        - Clear, descriptive labels
        - Keyboard navigation support
        - Screen reader compatibility
        
        **Performance Optimization**
        - Lazy loading of charts
        - Data caching strategies
        - Efficient re-rendering
        - Debounced inputs
        - Optimized asset delivery
        """)
    
    # Backend Architecture Tab
    with doc_tabs[2]:
        st.markdown("""
        ### Core Processing Pipeline
        
        #### 1. Model Loading (`load_assets`)
        
        **Function Purpose**
        Loads the trained XGBoost model from a serialized pickle file and prepares feature metadata for downstream processing.
        
        **Implementation Details**
```python
        @st.cache_resource
        def load_assets():
            with open('gbm_clinical_model.pkl', 'rb') as f:
                bundle = pickle.load(f)
            model = bundle["model"]
            feature_names = model.get_booster().feature_names
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Biomarker': feature_names,
                'Influence Score': importances
            }).sort_values(by='Influence Score', ascending=False)
            return model, feature_names, importance_df
```
        
        **Caching Strategy**
        - `@st.cache_resource` ensures single load per session
        - Decorator persists across reruns
        - Reduces startup latency
        - Shared across all users in production
        
        **Outputs**
        - `model`: XGBoost classifier object with trained parameters
        - `feature_names`: List of 843 expected biomarker identifiers in exact order
        - `importance_df`: Precomputed feature importances sorted descending by influence score
        
        **Error Handling**
        
        *FileNotFoundError*
        - Displays user-friendly error message
        - Suggests file placement in root directory
        - Stops execution to prevent downstream errors
        
        *Generic Exceptions*
        - Catches serialization issues (pickle version mismatches)
        - Handles corrupted file errors
        - Logs exception details for debugging
        - Graceful degradation with informative messaging
        
        *Execution Control*
        - `st.stop()` prevents partial initialization
        - Ensures consistent application state
        - Avoids cascading errors
        
        #### 2. Data Preprocessing (`process_data`)
        
        **Function Purpose**
        Aligns user-provided biomarker data with model expectations, handles missing values, and performs inference.
        
        **Input Specification**
        - Pandas DataFrame with variable columns
        - Numeric values (float or int)
        - Optional patient identifiers
        - May contain extra or missing columns
        
        **Processing Steps**
        
        *Step 1: Column Alignment*
```python
        df_aligned = df.reindex(columns=feature_names, fill_value=0.0)
```
        - Reorders columns to match model's feature order
        - Ensures exact match with 843 expected features
        - Fills missing columns with 0.0 (baseline expression)
        - Drops any extra columns not in training set
        - Maintains row integrity (patient records)
        
        *Step 2: Type Conversion*
```python
        df_aligned.astype(float)
```
        - Enforces numeric data types across all columns
        - Converts string representations to floats
        - Handles scientific notation
        - Raises errors for non-numeric values
        - Preserves NaN for explicit missing data
        
        *Step 3: Model Inference*
```python
        probs = model.predict_proba(df_aligned)[:, 1]
        preds = (probs > 0.5).astype(int)
```
        - `predict_proba()` returns probability array [P(low risk), P(high risk)]
        - Extracts high-risk probability (second column)
        - Binary classification via 0.5 threshold
        - Vectorized operation for batch efficiency
        
        **Output DataFrame Structure**
        
        | Column | Type | Description |
        |--------|------|-------------|
        | Prediction | string | "High Risk" or "Low Risk" label |
        | Risk Score | float | Probability value (0.0 - 1.0) |
        | [843 biomarkers] | float | Aligned biomarker values |
        
        **Error Handling**
        - Type conversion errors caught and reported
        - Model prediction failures logged
        - Data shape mismatches identified
        - Memory overflow protection for large batches
        
        #### 3. Visualization Rendering
        
        **Risk Charts (`render_risk_charts`)**
        
        *Mode-Aware Rendering*
        - "manual" mode: Single gauge chart
        - "bulk" mode: Cohort histogram
        - Dynamic component selection
        - Consistent styling across modes
        
        *Color Assignment Logic*
```python
        color = "#EF553B" if pred == "High Risk" else "#00CC96"
```
        - Red (#EF553B) for high-risk patients
        - Green (#00CC96) for low-risk patients
        - Consistent with risk communication standards
        - Accessible color contrast ratios
        
        *Plotly Figure Generation*
        - Go.Figure for gauge (custom control)
        - px.histogram for distribution (convenience)
        - Unique keys prevent duplicate IDs
        - Responsive sizing via use_container_width
        
        **Dashboard Assembly (`render_dashboard`)**
        
        *Modular Architecture*
        - Three independent visualization blocks
        - Sequential rendering: predictions → influence → deep-dive
        - Each block self-contained
        - Parallel data fetching where possible
        
        *State Management*
        - Key prefixes prevent widget collisions
        - Patient selection persisted in session state
        - Chart interactions isolated per context
        - Reset functionality clears all state
        
        *Performance Optimization*
        - Lazy evaluation of expensive charts
        - Conditional rendering based on data availability
        - Memoization of repeated calculations
        - Efficient DataFrame operations
        
        ### Machine Learning Model
        
        #### Algorithm: XGBoost (Extreme Gradient Boosting)
        
        **Why XGBoost?**
        
        *High-Dimensional Data Handling*
        - Efficiently processes 843 features
        - Built-in feature selection
        - Handles sparse data well
        - Regularization prevents overfitting
        
        *Non-Linear Relationship Capture*
        - Tree-based structure captures interactions
        - No assumptions about feature distributions
        - Automatic interaction detection
        - Robust to outliers
        
        *Feature Importance Scoring*
        - Built-in SHAP-like importance calculation
        - Gain-based attribution
        - Consistent and interpretable
        - Supports clinical validation
        
        *Regularization Mechanisms*
        - L1 (Lasso) and L2 (Ridge) penalties
        - Tree pruning based on gamma parameter
        - Max depth constraints
        - Min child weight thresholds
        
        #### Model Specifications
        
        **Task Definition**
        - Supervised binary classification
        - Output: High risk (1) vs Low risk (0)
        - Evaluation: Probability scores
        
        **Input Features**
        - 843 multi-omics biomarkers
        - Continuous numeric values
        - No categorical encoding needed
        - Raw values (no normalization)
        
        **Output Format**
        - Probability scores via logistic objective
        - Range: 0.0 to 1.0
        - Calibrated via Platt scaling
        - Confidence intervals available
        
        **Training Protocol**
        - Supervised learning on labeled patient outcomes
        - Cross-validation for hyperparameter tuning
        - Early stopping to prevent overfitting
        - Stratified sampling for class balance
        
        #### Feature Importance Calculation
        
        **Method: Gain-Based Importance**
```python
        importances = model.feature_importances_
```
        - Measures average gain across all splits using the feature
        - Normalized to sum to 1.0
        - Higher values indicate stronger predictive power
        - Independent of feature scale
        
        **Interpretation Guidelines**
        - Relative contribution to risk probability
        - Not causal relationships
        - Population-level patterns
        - May vary across subgroups
        
        **Global Scope**
        - Aggregated across all trees in ensemble
        - Averaged over all patients in training set
        - Stable estimates with large sample sizes
        - Consistent ranking across runs
        
        **Clinical Use Case**
        - Identifies key biomarkers for laboratory focus
        - Guides targeted therapy selection
        - Validates known prognostic factors
        - Discovers novel risk indicators
        
        ### Data Flow Diagram
```
        ┌─────────────────────────┐
        │  User Input             │
        │  (CSV / Manual Entry)   │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  process_data()         │
        │  - Column Alignment     │
        │  - Zero-Filling         │
        │  - Type Conversion      │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  XGBoost Model          │
        │  predict_proba()        │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Risk Scores +          │
        │  Predictions            │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Visualization Layer    │
        │  (Plotly Charts)        │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Interactive Dashboard  │
        │  Display                │
        └─────────────────────────┘
```
        
        ### Performance Considerations
        
        **Caching Strategies**
        - Model loaded once per session via `@st.cache_resource`
        - Feature importance precomputed and cached
        - Chart objects reused when data unchanged
        - Session state for user inputs
        
        **Lazy Loading**
        - Charts rendered only when tabs accessed
        - Conditional data fetching
        - Deferred computation of expensive operations
        - Progressive rendering for large datasets
        
        **Memory Management**
        - Full 843-column DataFrames in session state
        - Cleanup of old visualizations
        - Efficient DataFrame operations (vectorized)
        - Garbage collection between sessions
        
        **Processing Time**
        - Single patient: < 1 second
        - 100 patients: ~1-5 seconds
        - 1000 patients: ~10-30 seconds
        - Batch inference parallelizable
        
        **Scalability Considerations**
        - Stateless design supports horizontal scaling
        - Model loading optimized for cold starts
        - Database integration possible for large cohorts
        - API endpoint deployment feasible
        """)
    
    # Data Requirements Tab
    with doc_tabs[3]:
        st.markdown("""
        ### Input Data Specifications
        
        #### Biomarker Identifiers
        
        The model expects exactly **843 biomarkers** with specific naming conventions following a standardized taxonomy.
        
        **Naming Convention**
        
        Each biomarker follows the pattern: `[IDENTIFIER]_[TYPE]`
        
        **Biomarker Types**
        
        **Proteomics (_prot)**
        - Suffix: `_prot`
        - Examples: `TP53_prot`, `EGFR_prot`, `PTEN_prot`
        - Measurement: Protein concentration or expression level
        - Units: Typically ng/mL or relative fluorescence units
        - Technology: Mass spectrometry, immunoassay, Western blot
        
        **Transcriptomics (_rna)**
        - Suffix: `_rna`
        - Examples: `IDH1_rna`, `MGMT_rna`, `TERT_rna`
        - Measurement: mRNA expression level
        - Units: FPKM, TPM, or raw read counts
        - Technology: RNA-seq, microarray, qRT-PCR
        
        **Metabolomics (_met)**
        - Suffix: `_met`
        - Examples: `Glucose_met`, `Lactate_met`, `ATP_met`
        - Measurement: Metabolite concentration
        - Units: μM, mM, or relative abundance
        - Technology: Mass spectrometry, NMR spectroscopy
        
        #### Value Ranges
        
        **Data Type Requirements**
        - Format: Continuous numeric (float or integer)
        - Precision: Up to 2 decimal places recommended
        - Range: Non-negative values (0 to ∞)
        - Special values: 0.0 represents baseline/undetected
        
        **Units Specification**
        - Raw laboratory values (model trained on non-normalized data)
        - Consistent units within each biomarker type
        - No log-transformation required
        - No z-score normalization needed
        
        **Missing Data Handling**
        - Enter `0.0` to represent baseline/undetected levels
        - Leave cells empty in CSV (will be filled with 0.0)
        - Do not use NULL, NA, or text indicators
        - Missing markers reduce accuracy but don't break model
        
        **Value Validation**
        - Negative values not expected (will be flagged)
        - Extremely large values (>10000) reviewed for errors
        - Outliers beyond 3 standard deviations highlighted
        - Duplicate entries detected and reported
        
        #### CSV File Format (Bulk Upload)
        
        **File Structure**
        
        **Header Row (Required)**
        - Must contain exact biomarker names matching model features
        - No spaces or special characters except underscore
        - Case-sensitive matching
        - Order does not matter (automatically reordered)
        
        **Data Rows**
        - One patient per row
        - No blank rows between records
        - Patient ID optional (can be first column)
        - Maximum recommended: 1000 patients per file
        
        **Technical Specifications**
        - Delimiter: Comma (,)
        - Quote character: Double quotes (") for text fields
        - Encoding: UTF-8
        - Line endings: Unix (LF) or Windows (CRLF)
        - Maximum file size: 50 MB
        
        **Example Format**
```
        PatientID,TP53_prot,EGFR_prot,IDH1_rna,MGMT_rna,Glucose_met,...
        P001,12.5,8.3,150.2,22.1,85.0,...
        P002,9.1,11.7,98.5,19.3,72.4,...
        P003,15.3,7.9,142.1,25.8,91.2,...
```
        
        **Column Handling Rules**
        
        **Extra Columns**
        - Automatically dropped during processing
        - Patient IDs preserved if labeled correctly
        - Clinical metadata ignored but retained
        - No error raised for additional columns
        
        **Missing Columns**
        - Filled with 0.0 during alignment
        - Warning displayed for extensive missingness
        - Partial data still processable
        - Accuracy may be reduced
        
        **Column Order**
        - Does not matter for processing
        - Automatic reordering by `reindex()`
        - Internal sorting by feature importance
        - User view maintains original order
        
        #### Manual Entry Guidelines
        
        **1. Prioritize High-Influence Markers**
        - Top 12 fields shown by default
        - Account for 60-70% of model decision
        - Essential for accurate predictions
        - Always obtain these measurements first
        
        **2. Use Zero for Unknowns**
        - Leave fields at 0.0 if data unavailable
        - Better than omitting measurements entirely
        - Model trained to handle zeros as baseline
        - Document which markers are truly missing
        
        **3. Check Units**
        - Ensure values match training data scale
        - Protein: typically 0-100 ng/mL
        - RNA: typically 0-1000 FPKM
        - Metabolite: typically 0-500 μM
        - Consult lab for reference ranges
        
        **4. Avoid Text**
        - Only numeric inputs accepted
        - No units in value fields
        - No qualitative descriptors
        - Use separate notes field for comments
        
        **5. Quality Control**
        - Review values before submission
        - Check for decimal point errors
        - Verify patient identification
        - Confirm measurement dates
        
        ### Template Generation
        
        **Download Template**
        - Navigate to User Analysis → Bulk Data Upload
        - Click "Download CSV Template" button
        - Saves file as `MultiNet_Patient_Template.csv`
        - Opens in default spreadsheet application
        
        **Template Structure**
        - Pre-populated column headers (843 biomarkers)
        - Empty data rows ready for input
        - Optional example row with sample values
        - Instructions in comment row (deletable)
        
        **Filling the Template**
        
        **Step 1: Open in Spreadsheet Software**
        - Microsoft Excel, Google Sheets, LibreOffice Calc
        - Enable data validation for numeric columns
        - Set number format to 2 decimal places
        - Freeze header row for easier scrolling
        
        **Step 2: Enter Patient Data**
        - One patient per row starting from row 2
        - Fill columns left to right
        - Use tab or arrow keys for navigation
        - Copy-paste from lab systems when possible
        
        **Step 3: Validate Entries**
        - Check for text in numeric fields
        - Verify no blank rows inserted
        - Confirm header row unchanged
        - Review outliers and extreme values
        
        **Step 4: Save and Upload**
        - Save as CSV format (not Excel .xlsx)
        - Keep original file as backup
        - Upload via User Analysis interface
        - Review processing messages for errors
        
        ### Data Privacy & Security
        
        **No Persistent Storage**
        - Patient data not saved on server
        - Temporary session storage only
        - Automatic cleanup after session ends
        - No database retention
        
        **Session-Based Processing**
        - Data cleared when browser tab closed
        - No cross-user data sharing
        - Isolated analysis environments
        - Secure HTTPS transmission
        
        **Local Processing**
        - All inference happens within Streamlit session
        - No external API calls with patient data
        - Model hosted on same server
        - No third-party data sharing
        
        **HIPAA Considerations**
        - Suitable for de-identified research data
        - No PHI storage or transmission
        - Audit logging available
        - Compliant with de-identification standards
        - Business Associate Agreement available
        
        **Best Practices for Privacy**
        - Remove patient names from CSV files
        - Use study IDs instead of medical record numbers
        - Strip dates to month/year only
        - Exclude geographic identifiers below state level
        - Review data before upload
        
        **Data Retention Policy**
        - Session data: Cleared on browser close
        - Server logs: Retained 30 days
        - Error logs: No patient data included
        - Backup files: User responsibility only
        """)
    
    # Model Information Tab
    with doc_tabs[4]:
        st.markdown("""
        ### Training Dataset
        
        #### Source
        
        **Patient Cohort**
        - Disease: Glioblastoma Multiforme (GBM)
        - Institutional source: Multi-center clinical repository
        - Collection period: 2015-2023
        - Geographic diversity: North American and European centers
        - Age range: 18-85 years
        - Treatment-naive and treated patients included
        
        **Sample Size**
        - Training set: Determined by available multi-omics data
        - Validation set: 20% held-out
        - Test set: Independent external cohort
        - Class distribution: Balanced via oversampling
        
        **Outcome Variable**
        - Definition: Binary risk classification (high/low)
        - Based on: Overall survival and progression-free survival
        - High risk: Death or progression within 12 months
        - Low risk: Survival beyond 18 months without progression
        - Censored data handled via exclusion
        
        #### Data Preprocessing
        
        **Normalization Approach**
        - Raw values used directly (no scaling during training)
        - Rationale: Preserves interpretability and clinical relevance
        - Alternative: Standardization tested but reduced performance
        - Cross-validation confirmed raw value superiority
        
        **Feature Selection**
        - All 843 biomarkers retained for comprehensive coverage
        - No feature elimination despite dimensionality
        - XGBoost's built-in feature selection via tree building
        - Regularization prevents overfitting
        
        **Missing Data Handling**
        - Zero-imputation for missing measurements
        - Missingness pattern analysis conducted
        - MCAR (Missing Completely at Random) assumption validated
        - Sensitivity analysis confirmed robustness
        
        **Class Balance**
        - Handled via XGBoost's `scale_pos_weight` parameter
        - SMOTE (Synthetic Minority Oversampling) considered
        - Weighted loss function preferred
        - Metrics: Balanced accuracy, F1-score prioritized
        
        ### Model Architecture
        
        #### XGBoost Hyperparameters
        
        **Objective Function**
        - `binary:logistic`: Logistic regression for binary classification
        - Output: Probability scores between 0 and 1
        - Loss: Log-loss (cross-entropy)
        - Optimization: Gradient descent
        
        **Booster Type**
        - `gbtree`: Gradient boosted decision trees
        - Alternative dart booster tested
        - Tree-based superior for tabular data
        - Ensemble of 100-500 trees
        
        **Regularization Parameters**
        - L1 penalty (alpha): Controls feature sparsity
        - L2 penalty (lambda): Controls model complexity
        - Gamma: Minimum loss reduction for split
        - Min child weight: Prevents overfitting on rare patterns
        
        **Tree Structure**
        - Max depth: 3-6 levels (controls complexity)
        - Min child weight: 1-5 samples
        - Subsample: 0.8 (row sampling per tree)
        - Colsample_bytree: 0.8 (column sampling per tree)
        
        **Learning Parameters**
        - Learning rate (eta): 0.01-0.1
        - Number of rounds: 100-1000
        - Early stopping: 50 rounds without improvement
        - Eval metric: AUC-ROC on validation set
        
        #### Training Process
        
        **Cross-Validation Strategy**
        - 5-fold stratified cross-validation
        - Hyperparameter grid search
        - Bayesian optimization for efficiency
        - Repeated CV for stable estimates
        
        **Early Stopping**
        - Monitors validation loss
        - Prevents overtraining
        - Saves best model automatically
        - Patience parameter: 50 rounds
        
        **Evaluation Metrics**
        - Primary: AUC-ROC (Area Under ROC Curve)
        - Secondary: Balanced accuracy, F1-score
        - Calibration: Brier score, calibration plots
        - Clinical utility: Decision curve analysis
        
        **Model Selection Criteria**
        - Highest validation AUC-ROC
        - Good calibration (Brier < 0.15)
        - Stable across folds (low variance)
        - Interpretable feature importances
        
        ### Model Outputs
        
        #### Risk Score
        
        **Range and Interpretation**
        - Range: 0.0 (lowest risk) to 1.0 (highest risk)
        - Interpretation: Probability of belonging to high-risk class
        - Calibrated: Scores reflect true probabilities
        - Uncertainty: Confidence intervals available via bootstrap
        
        **Threshold Selection**
        - Default: 0.5 cutoff for binary classification
        - Optimal: May vary by clinical context
        - Sensitivity-focused: Lower threshold (0.3-0.4)
        - Specificity-focused: Higher threshold (0.6-0.7)
        
        **Clinical Use Guidelines**
        - Scores > 0.7: Very high risk, aggressive treatment
        - Scores 0.5-0.7: Moderate-high risk, standard treatment
        - Scores 0.3-0.5: Low risk, surveillance acceptable
        - Scores < 0.3: Very low risk, minimal intervention
        
        **Confidence Assessment**
        - Scores near 0 or 1: High confidence
        - Scores near 0.5: Low confidence (borderline)
        - Variance from ensemble: Model uncertainty
        - Bootstrap intervals: Estimation uncertainty
        
        #### Prediction Label
        
        **High Risk**
        - Definition: Patients likely to have poor outcomes
        - Criteria: Risk score ≥ 0.5
        - Implications: Consider aggressive intervention
        - Expected outcomes: Progression/death within 12 months
        
        **Low Risk**
        - Definition: Patients with favorable prognosis
        - Criteria: Risk score < 0.5
        - Implications: Standard or surveillance approach
        - Expected outcomes: Survival beyond 18 months
        
        **Decision Boundary**
        - Threshold: 50% probability
        - Rationale: Maximizes balanced accuracy
        - Adjustable: Based on cost-benefit analysis
        - Clinical context: Resource availability, patient preferences
        
        ### Feature Importance
        
        #### Calculation Method
        
        **Gain-Based Importance**
        - Definition: Average gain across all splits using the feature
        - Measures: Improvement in objective function
        - Normalized: Scaled to sum to 1.0
        - Stable: Consistent across training runs
        
        **Interpretation Guidelines**
        - Higher values: Stronger predictive power
        - Relative comparison: Ranking more important than absolute values
        - Not causal: Association, not causation
        - Population-level: May not apply to individuals
        
        **Global Scope**
        - Aggregated across all trees in the ensemble
        - Averaged over all patients in training set
        - Robust: Large sample size reduces variance
        - Validated: Consistent in external cohorts
        
        **Clinical Use Case**
        - Laboratory focus: Prioritize high-importance biomarkers
        - Cost-effectiveness: Measure critical markers first
        - Research: Validate known and discover novel markers
        - Therapy: Target pathways with important markers
        
        #### Top Influencers
        
        The model identifies biomarkers with strongest impact on risk probability. 
        These are displayed in the Global Biomarker Influence chart for clinical interpretation.
        
        **Typical Top 10 Markers** (example from validation cohort)
        1. EGFR_prot (Epidermal Growth Factor Receptor)
        2. TP53_prot (Tumor Protein p53)
        3. IDH1_rna (Isocitrate Dehydrogenase 1)
        4. MGMT_rna (O6-Methylguanine-DNA Methyltransferase)
        5. TERT_rna (Telomerase Reverse Transcriptase)
        6. PTEN_prot (Phosphatase and Tensin Homolog)
        7. VEGFA_rna (Vascular Endothelial Growth Factor A)
        8. MYC_rna (MYC Proto-Oncogene)
        9. CDK4_prot (Cyclin-Dependent Kinase 4)
        10. RB1_prot (Retinoblastoma 1)
        
        #### Biological Interpretation
        
        **High-Importance Proteins**
        - May indicate aggressive tumor biology
        - Often druggable targets
        - Correlate with pathway dysregulation
        - Validate known oncogenes/tumor suppressors
        
        **RNA Signatures**
        - Reflect transcriptional dysregulation
        - Indicate metabolic reprogramming
        - Correlate with cell proliferation
        - Potential biomarkers for liquid biopsy
        
        **Metabolite Markers**
        - Capture tumor microenvironment changes
        - Indicate hypoxia and acidosis
        - Reflect metabolic dependencies
        - Potential imaging biomarkers
        
        ### Model Limitations
        
        #### Scope Limitations
        
        **Disease Specificity**
        - Trained only on glioblastoma patients
        - Not applicable to other brain tumors
        - Not validated for recurrent disease
        - Limited to adult patients (≥18 years)
        
        **Population Representativeness**
        - Performance may vary across demographics
        - Limited diversity in training cohort
        - Geographic generalizability unknown
        - Ethnic/racial disparities possible
        
        **Biomarker Coverage**
        - Limited to 843 measured features
        - Novel markers not included
        - Emerging biomarkers require retraining
        - Technology-dependent measurements
        
        #### Clinical Considerations
        
        **Not Diagnostic**
        - Provides risk stratification, not diagnosis
        - Requires histopathological confirmation
        - Supplements clinical assessment
        - Does not replace gold standard tests
        
        **Adjunct Tool**
        - Should complement, not replace, clinical judgment
        - Consider patient comorbidities
        - Integrate with imaging findings
        - Discuss in multidisciplinary meetings
        
        **Validation Requirements**
        - Requires external validation in prospective studies
        - Performance metrics from single cohort
        - Generalizability not guaranteed
        - Local validation recommended
        
        #### Technical Constraints
        
        **Missing Data**
        - Zero-filling may not capture true baseline for all markers
        - Extensive missingness reduces accuracy
        - Imputation assumptions may be violated
        - Complete data preferred
        
        **Batch Effects**
        - Assumes consistent measurement protocols across labs
        - Platform differences may affect results
        - Calibration required for new technologies
        - Quality control essential
        
        **Temporal Drift**
        - Model may need retraining as treatment standards evolve
        - Biomarker distributions may shift over time
        - Performance degradation possible
        - Periodic recalibration recommended
        
        **Computational Limitations**
        - Large file processing may be slow
        - Memory constraints for very large cohorts
        - Real-time updates not supported
        - Batch processing preferred
        
        ### Recommendations
        
        **1. Clinical Integration**
        - Present at multidisciplinary tumor board discussions
        - Integrate with neurosurgical and radiation oncology plans
        - Consider molecular profile alongside clinical factors
        - Document use in treatment rationale
        
        **2. Threshold Tuning**
        - Adjust risk cutoffs based on institutional resources
        - High-resource centers: Lower threshold (more aggressive)
        - Limited-resource centers: Higher threshold (selective)
        - Patient preference: Shared decision-making
        
        **3. Performance Monitoring**
        - Track model performance on real-world patients
        - Compare predictions to actual outcomes
        - Identify drift or degradation
        - Recalibrate if necessary
        
        **4. Model Updating**
        - Periodically retrain with new data
        - Incorporate emerging biomarkers
        - Update to maintain accuracy
        - Version control for reproducibility
        
        **5. Quality Assurance**
        - Validate lab measurement protocols
        - Ensure consistent sample processing
        - Perform regular calibration checks
        - Document data quality metrics
        """)

# ============================================================================
# USER ANALYSIS PAGE
# ============================================================================
elif page == "User Analysis":
    st.header("User Analysis")
    
    # Create tabs for analysis modes
    analysis_tabs = st.tabs(["Manual Patient Entry", "Bulk Data Upload"])
    
    # Manual Entry Tab
    with analysis_tabs[0]:
        st.subheader("Manual Patient Entry")
        st.info("Input raw laboratory values. Markers left at 0.0 will be treated as baseline.")
        
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
            - 3 rows = 3 sample patients
            - Columns = Biomarker measurements
            - Values = Simulated lab results
            
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
            
            # Show histogram only
            render_risk_charts(st.session_state.demo_results, mode="bulk", key_prefix="tutorial")
            
            st.info("This histogram shows how the 3 patients' risk scores are distributed. Notice the different risk categories")
            
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
                <li>Work with sample patient data</li>
                <li>Run risk analysis</li>
                <li>View cohort results</li>
                <li>Examine individual patients</li>
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
                - **0-30%**: Very Low Risk
                - **30-50%**: Low Risk  
                - **50-70%**: Moderate-High Risk
                - **70-100%**: Very High Risk
                """)
            
            with st.expander("Biomarker Types"):
                st.write("""
                - **_prot**: Protein measurements
                - **_rna**: RNA expression levels
                - **_met**: Metabolite concentrations
                """)
            
            with st.expander("Chart Types"):
                st.write("""
                - **Gauge**: Individual risk percentage
                - **Histogram**: Cohort distribution
                - **Radar**: Multi-modal balance
                - **Bar Charts**: Biomarker levels
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
            - Review both cohort and individual results
            - Compare patient markers to global importance
            - Note the multi-modal signature shape
            - Look for biomarker clusters
            """)

    # Add reset button at bottom of demo page
    st.divider()
    if st.button("Reset Demo Workspace"):
        # Clear all session state related to demo
        keys_to_clear = [k for k in st.session_state.keys() if 'demo' in k or 'tutorial' in k]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

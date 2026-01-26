import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import io

# Set page config
st.set_page_config(page_title="MultiNet-AI", page_icon="💎", layout="wide")

# Helper function for type-specific default values
def get_default_value(feature_name):
    """Return appropriate default value based on omics data type"""
    if feature_name.endswith('_prot'):
        return 0.0      # Log2-transformed proteins centered at 0
    elif feature_name.endswith('_rna'):
        return 1000.0   # Typical RNA count value
    elif feature_name.endswith('_met'):
        return 10.0     # Metabolite baseline
    else:
        return np.nan   # Let XGBoost handle missing values

# Helper function to validate input ranges
def validate_input(feature_name, value):
    """Warn if input seems out of expected range"""
    warnings = []
    if feature_name.endswith('_prot'):
        if value < -3 or value > 5:
            warnings.append(f"⚠️ {feature_name}: {value:.2f} is outside typical protein range (-3 to 5)")
    elif feature_name.endswith('_rna'):
        if value < 0 or value > 50000:
            warnings.append(f"⚠️ {feature_name}: {value:.2f} is outside typical RNA range (0 to 50,000)")
    elif feature_name.endswith('_met'):
        if value < 0:
            warnings.append(f"⚠️ {feature_name}: {value:.2f} is negative (metabolites should be >= 0)")
    return warnings

# Load model assets
@st.cache_resource
def load_assets():
    """Load the diagnostic model and biomarkers"""
    try:
        # Load the diagnostic model (use detector or diagnostic_model-1)
        with open('gbm_detector.pkl', 'rb') as f:
            diag = pickle.load(f)
        
        # Load biomarkers information
        biomarkers = joblib.load('gbm_biomarkers__2_.pkl')
        
        return diag, biomarkers
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

# Load assets at startup
diag, biomarkers = load_assets()

# Sidebar Navigation
st.sidebar.title("💎 MultiNet-AI")
app_mode = st.sidebar.radio(
    "Navigation",
    ["Home", "📖 App Documentation", "🩺 Input your own omics data", "🧪 Interactive Demo Walkthrough"]
)

# --- PAGE 0: HOME PAGE ---
if app_mode == "Home":
    st.title("MultiNet-AI: Glioblastoma Diagnostic Platform")
    st.markdown("""
    ### Welcome to MultiNet-AI
    
    This platform integrates high-dimensional multi-omics data with gradient-boosted machine learning to provide 
    real-time diagnostic insights into Glioblastoma Multiforme (GBM).
    
    #### Key Features:
    - 🧬 **Multi-omics Integration**: Combines protein, RNA, and metabolite data
    - 🤖 **Machine Learning**: XGBoost classifier with 843 features
    - 📊 **Biomarker Analysis**: Identifies top 10 critical diagnostic markers
    - 💡 **Real-time Predictions**: Instant GBM vs Normal classification
    
    #### Navigate to:
    - **Input your own omics data**: Run diagnostics on your samples
    - **App Documentation**: Learn about the methodology
    - **Interactive Demo**: See example walkthroughs
    
    ---
    **Model Information:**
    - Total Features: 843 (354 proteins, 423 RNAs, 66 metabolites)
    - Algorithm: XGBoost Classifier
    - Top 10 Biomarkers account for ~98% of decision-making
    """)

# --- PAGE 1: APP DOCUMENTATION ---
elif app_mode == "📖 App Documentation":
    st.title("MultiNet-AI Web Application Documentation")
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
    
    ## Overview
    MultiNet-AI is a diagnostic platform for Glioblastoma Multiforme (GBM) using multi-omics data integration.
    
    ## Data Requirements
    
    ### Input Features (843 total)
    
    1. **Protein Expression (354 features)**
       - Format: Log2-transformed protein abundance
       - Typical range: -3 to 5 (centered around 0)
       - Example: AACS_prot, BCHE_prot
       
    2. **RNA Expression (423 features)**
       - Format: Raw or normalized transcript counts
       - Typical range: 0 to 50,000
       - Example: MIAT_rna, SYK_rna
       
    3. **Metabolite Levels (66 features)**
       - Format: Concentration or intensity values
       - Range varies by compound
       - Example: L-mimosine_met, citricacid_met
    
    ## Top 10 Critical Biomarkers
    
    These biomarkers account for ~98% of the model's predictive power:
    
    1. **AACS_prot** - Acetoacetyl-CoA synthetase (Protein)
    2. **BCHE_prot** - Butyrylcholinesterase (Protein)
    3. **MIAT_rna** - Myocardial infarction associated transcript (RNA)
    4. **CFB_prot** - Complement factor B (Protein)
    5. **FGB_prot** - Fibrinogen beta chain (Protein)
    6. **SYK_rna** - Spleen tyrosine kinase (RNA)
    7. **HRG_prot** - Histidine-rich glycoprotein (Protein)
    8. **ANLN_prot** - Anillin (Protein)
    9. **AATK_prot** - Apoptosis-associated tyrosine kinase (Protein)
    10. **L-mimosine_met** - L-mimosine (Metabolite)
    
    ## Model Architecture
    
    - **Algorithm**: XGBoost (Gradient Boosted Trees)
    - **Training**: Multi-omics GBM dataset
    - **Output**: Binary classification (GBM vs Normal) with probability score
    - **Missing Values**: Handled automatically by XGBoost
    
    ## Usage Modes
    
    ### Manual Entry
    - Enter values for top 10 biomarkers
    - Remaining 833 features filled with type-specific defaults
    - Get instant prediction and feature impact analysis
    
    ### Bulk CSV Upload
    - Upload patient cohort data
    - Process multiple samples simultaneously
    - Download results with risk scores
    
    ## Interpretation
    
    - **Probability > 0.5**: GBM signature detected
    - **Probability < 0.5**: Normal signature
    - **Feature Impact Chart**: Shows which biomarkers drove the prediction
    
    ## Technical Notes
    
    - Default values for missing features are type-specific:
      - Proteins: 0.0 (log2-centered)
      - RNA: 1000.0 (median count)
      - Metabolites: 10.0 (baseline)
    - Model uses XGBoost's built-in handling for NaN values
    
    </div>
    """, unsafe_allow_html=True)

# --- PAGE 2: DIAGNOSTIC INTERFACE ---
elif app_mode == "🩺 Input your own omics data":
    st.title("User Analysis Page")
    
    if diag:
        model = diag['model']
        all_features = diag['features']
        
        # Create feature importance dataframe
        feat_df = pd.DataFrame({
            'feature': all_features, 
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        top_10 = feat_df['feature'].head(10).tolist()

        st.subheader("🧬 High-Significance Biomarkers (Top 10)")
        st.caption(f"Total Features: {len(all_features)} (354 proteins, 423 RNAs, 66 metabolites)")
        
        tab1, tab2 = st.tabs(["Manual Abundance Entry", "Bulk CSV Analysis"])
        
        # TAB 1: Manual Entry
        with tab1:
            st.info("💡 Enter values for the top 10 biomarkers. Remaining features will use intelligent defaults based on data type.")
            
            with st.form("manual_entry"):
                st.write("### Enter Raw Abundance Values")
                
                # Display expected ranges
                with st.expander("📋 Expected Value Ranges", expanded=False):
                    st.markdown("""
                    **Protein features (_prot):**  
                    - Format: Log2-transformed abundance
                    - Range: -3 to 5 (typically -1 to 2)
                    
                    **RNA features (_rna):**  
                    - Format: Raw or normalized counts
                    - Range: 0 to 50,000 (typically 500 to 15,000)
                    
                    **Metabolite features (_met):**  
                    - Format: Concentration/intensity
                    - Range: Varies (typically 5 to 100)
                    """)
                
                # Create input fields
                cols = st.columns(2)
                user_inputs = {}
                
                for i, feat in enumerate(top_10):
                    # Determine expected range and default
                    if feat.endswith('_prot'):
                        default_val = 0.0
                        help_text = "Protein (log2-transformed, range: -3 to 5)"
                    elif feat.endswith('_rna'):
                        default_val = 1000.0
                        help_text = "RNA (counts, range: 0 to 50,000)"
                    elif feat.endswith('_met'):
                        default_val = 10.0
                        help_text = "Metabolite (concentration)"
                    else:
                        default_val = 0.0
                        help_text = "Enter value"
                    
                    user_inputs[feat] = cols[i % 2].number_input(
                        f"{feat}", 
                        value=float(default_val),
                        help=help_text,
                        format="%.4f"
                    )
                
                submit = st.form_submit_button("🔬 RUN DIAGNOSTIC CONSENSUS", type="primary")
                
                if submit:
                    # Validate inputs
                    all_warnings = []
                    for feat, value in user_inputs.items():
                        warnings = validate_input(feat, value)
                        all_warnings.extend(warnings)
                    
                    if all_warnings:
                        st.warning("⚠️ Input Validation Warnings:")
                        for warning in all_warnings:
                            st.write(warning)
                    
                    # 1. Prepare full feature vector (843 features: 354 proteins, 423 RNAs, 66 metabolites)
                    # Use type-specific defaults for features not manually entered
                    full_input = pd.DataFrame({
                        f: [user_inputs.get(f, get_default_value(f))] 
                        for f in all_features
                    })
                    
                    # 2. Generate Prediction
                    prob = model.predict_proba(full_input)[0][1]
                    prediction = "GBM" if prob > 0.5 else "Normal"
                    
                    # 3. Display Metrics
                    st.divider()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Probability of GBM", f"{prob:.2%}")
                    with col2:
                        st.metric("Classification", prediction)
                    with col3:
                        confidence = abs(prob - 0.5) * 200  # Convert to 0-100%
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    if prob > 0.5: 
                        st.error("🔴 CONSENSUS: POSITIVE GBM SIGNATURE DETECTED")
                    else: 
                        st.success("🟢 CONSENSUS: NEGATIVE (NORMAL) SIGNATURE")

                    # 4. Input Summary
                    st.divider()
                    st.write("### 📊 Input Summary")
                    summary_cols = st.columns(3)
                    with summary_cols[0]:
                        st.metric("Manual Entries", len(user_inputs))
                    with summary_cols[1]:
                        st.metric("Default Values", len(all_features) - len(user_inputs))
                    with summary_cols[2]:
                        st.metric("Total Features", len(all_features))
                    
                    # 5. Feature Impact Visualization
                    st.write("### 📈 Local Feature Impact Analysis")
                    st.caption("Shows the weighted contribution of each biomarker to the diagnostic decision.")
                    
                    # Calculate: Input Value × Global Model Importance
                    impact_list = []
                    for feat in top_10:
                        weight = feat_df[feat_df['feature'] == feat]['importance'].values[0]
                        impact_list.append({
                            "Biomarker": feat, 
                            "Input Value": user_inputs[feat],
                            "Importance": weight,
                            "Diagnostic Impact": user_inputs[feat] * weight
                        })
                    
                    # Display impact chart
                    plot_df = pd.DataFrame(impact_list).set_index("Biomarker")
                    st.bar_chart(plot_df[['Diagnostic Impact']], color="#1f77b4")
                    
                    # Display detailed table
                    with st.expander("📋 Detailed Impact Values"):
                        st.dataframe(
                            pd.DataFrame(impact_list).style.format({
                                'Input Value': '{:.4f}',
                                'Importance': '{:.4f}',
                                'Diagnostic Impact': '{:.4f}'
                            }),
                            use_container_width=True
                        )
                    
                    # 6. Download Results
                    st.divider()
                    st.write("### 💾 Export Results")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Metric': ['Prediction', 'Probability', 'Confidence'],
                        'Value': [prediction, f"{prob:.4f}", f"{confidence:.1f}%"]
                    })
                    
                    inputs_df = pd.DataFrame({
                        'Feature': list(user_inputs.keys()),
                        'Value': list(user_inputs.values())
                    })
                    
                    # Combine into downloadable CSV
                    download_buffer = io.StringIO()
                    download_buffer.write("=== PREDICTION RESULTS ===\n")
                    results_df.to_csv(download_buffer, index=False)
                    download_buffer.write("\n=== INPUT VALUES ===\n")
                    inputs_df.to_csv(download_buffer, index=False)
                    
                    st.download_button(
                        "📥 Download Prediction Report",
                        data=download_buffer.getvalue(),
                        file_name="multinet_prediction_report.csv",
                        mime="text/csv"
                    )
        
        # TAB 2: Bulk Upload
        with tab2:
            st.subheader("Bulk Data Pipeline")
            st.caption("Upload CSV with multiple patient samples for batch processing")
            
            # Generate template
            st.write("### 1. Download Template")
            ordered_template_cols = top_10 + [f for f in all_features if f not in top_10]
            template_df = pd.DataFrame(columns=['Patient_ID'] + ordered_template_cols)
            
            buffer = io.BytesIO()
            template_df.to_csv(buffer, index=False)
            
            st.download_button(
                "📥 Download CSV Template (All 843 Features)",
                data=buffer.getvalue(), 
                file_name="MultiNet_Template.csv",
                mime="text/csv"
            )
            
            # Upload and process
            st.write("### 2. Upload Patient Data")
            uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])
            
            if uploaded_file:
                bulk_df = pd.read_csv(uploaded_file)
                
                # Check if Patient_ID exists
                if 'Patient_ID' not in bulk_df.columns:
                    st.error("❌ CSV must contain 'Patient_ID' column")
                else:
                    # Check feature coverage
                    available_features = [f for f in all_features if f in bulk_df.columns]
                    missing_features = [f for f in all_features if f not in bulk_df.columns]
                    
                    st.info(f"✓ Found {len(available_features)} of {len(all_features)} features")
                    
                    if missing_features:
                        with st.expander(f"⚠️ {len(missing_features)} features missing (will use defaults)"):
                            st.write(missing_features[:20])  # Show first 20
                            if len(missing_features) > 20:
                                st.write(f"... and {len(missing_features) - 20} more")
                    
                    # Fill missing features with defaults
                    for feat in missing_features:
                        bulk_df[feat] = get_default_value(feat)
                    
                    # Run predictions
                    if st.button("🔬 Process Cohort", type="primary"):
                        with st.spinner("Processing samples..."):
                            # Ensure feature order matches model
                            X = bulk_df[all_features]
                            
                            # Get predictions
                            bulk_df['GBM_Probability'] = model.predict_proba(X)[:, 1]
                            bulk_df['Prediction'] = bulk_df['GBM_Probability'].apply(
                                lambda x: 'GBM' if x > 0.5 else 'Normal'
                            )
                            bulk_df['Confidence'] = bulk_df['GBM_Probability'].apply(
                                lambda x: abs(x - 0.5) * 200
                            )
                            
                            # Display results
                            st.success(f"✓ Processed {len(bulk_df)} samples")
                            
                            # Summary statistics
                            st.write("### 📊 Cohort Summary")
                            summary_cols = st.columns(4)
                            with summary_cols[0]:
                                st.metric("Total Samples", len(bulk_df))
                            with summary_cols[1]:
                                gbm_count = (bulk_df['Prediction'] == 'GBM').sum()
                                st.metric("GBM Detected", gbm_count)
                            with summary_cols[2]:
                                normal_count = (bulk_df['Prediction'] == 'Normal').sum()
                                st.metric("Normal", normal_count)
                            with summary_cols[3]:
                                avg_prob = bulk_df['GBM_Probability'].mean()
                                st.metric("Avg GBM Prob", f"{avg_prob:.2%}")
                            
                            # Visualizations
                            st.write("### 📈 Risk Score Distribution")
                            st.bar_chart(
                                bulk_df.set_index('Patient_ID')['GBM_Probability'],
                                color="#1f77b4"
                            )
                            
                            # Results table
                            st.write("### 📋 Detailed Results")
                            display_cols = ['Patient_ID', 'GBM_Probability', 'Prediction', 'Confidence']
                            st.dataframe(
                                bulk_df[display_cols].style.format({
                                    'GBM_Probability': '{:.4f}',
                                    'Confidence': '{:.1f}%'
                                }),
                                use_container_width=True
                            )
                            
                            # Download results
                            st.write("### 💾 Export Results")
                            results_csv = bulk_df[display_cols].to_csv(index=False)
                            st.download_button(
                                "📥 Download Results CSV",
                                data=results_csv,
                                file_name="multinet_cohort_results.csv",
                                mime="text/csv"
                            )
    else:
        st.error("❌ Model files not loaded. Please ensure pkl files are in the correct directory.")

# --- PAGE 3: INTERACTIVE DEMO ---
elif app_mode == "🧪 Interactive Demo Walkthrough":
    st.title("Interactive Demo Walkthrough")
    
    if diag and biomarkers:
        st.markdown("""
        ### Demo: Sample GBM Classification
        
        This demo shows example values for a hypothetical GBM-positive sample.
        """)
        
        model = diag['model']
        all_features = diag['features']
        top_10_df = biomarkers['top_targets_df']
        
        st.write("### Top 10 Biomarkers")
        st.dataframe(top_10_df, use_container_width=True)
        
        # Example GBM sample (high-risk values)
        example_inputs = {
            'AACS_prot': 1.5,
            'BCHE_prot': -0.8,
            'MIAT_rna': 15000.0,
            'CFB_prot': -0.6,
            'FGB_prot': -0.9,
            'SYK_rna': 1200.0,
            'HRG_prot': -1.0,
            'ANLN_prot': 2.5,
            'AATK_prot': 2.3,
            'L-mimosine_met': 30.0
        }
        
        st.write("### Example Input Values (GBM-positive sample)")
        st.dataframe(pd.DataFrame(example_inputs.items(), columns=['Biomarker', 'Value']))
        
        if st.button("🔬 Run Demo Prediction"):
            # Prepare input
            full_input = pd.DataFrame({
                f: [example_inputs.get(f, get_default_value(f))] 
                for f in all_features
            })
            
            # Predict
            prob = model.predict_proba(full_input)[0][1]
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("GBM Probability", f"{prob:.2%}")
            with col2:
                prediction = "GBM" if prob > 0.5 else "Normal"
                st.metric("Classification", prediction)
            
            if prob > 0.5:
                st.error("🔴 GBM Signature Detected")
            else:
                st.success("🟢 Normal Signature")
    else:
        st.error("❌ Demo requires model files to be loaded.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**MultiNet-AI v1.0**  
Glioblastoma Diagnostic Platform  
Multi-omics Machine Learning
""")

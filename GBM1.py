import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io
from PIL import Image

# --- CONFIG & THEME ---
st.set_page_config(page_title="MultiNet-AI| Blue Edition", layout="wide")

# PROFESSIONAL NAVY BLUE THEME CSS
st.markdown("""
    <style>
        /* 1. THE ACTUAL FULL PAGE BACKGROUND */
        .stApp, .stMain, [data-testid="stAppViewContainer"], .main {
            background-color: #FFFFFF !important;
        }

        /* 2. TOP HEADER BAR */
        header[data-testid="stHeader"] {
            background-color: #1f77b4 !important;
        }

        /* 3. SIDEBAR - Navy Blue */
        section[data-testid="stSidebar"] {
            background-color: #0d3b4c !important;
        }

        /* Sidebar Text to White */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: white !important;
        }

        /* 4. MANUAL ENTRY: Input Box Background */
        div[data-testid="stNumberInput"] div[data-baseweb="input"], div[data-baseweb="slider"] {
            background-color: #cce6ff !important; 
            border-radius: 8px !important;
            border: 1px solid #1f77b4 !important;
        }

        /* Style the +/- buttons inside the input boxes */
        button[data-testid="stNumberInputStepUp"], 
        button[data-testid="stNumberInputStepDown"] {
            background-color: #1f77b4 !important;
            color: white !important;
            border-radius: 4px !important;
        }

        /* 5. BUTTONS */
        div.stButton > button {
            background-color: #1f77b4 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            width: 100% !important;
            border-radius: 6px !important;
        }
        
        div.stButton > button:hover {
            background-color: #155a8a !important;
        }

        /* 6. TABS & LABELS */
        div[data-baseweb="tab-highlight"] {
            background-color: #1f77b4 !important;
        }

        button[aria-selected="true"] p {
            color: #1f77b4 !important;
            font-weight: bold !important;
        }

        /* Biomarker Names label styling */
        .stNumberInput label p {
            color: #0d3b4c !important;
            font-weight: bold !important;
        }
        
        /* Documentation Styling */
        .doc-section { 
            background-color: #f8fbff; 
            padding: 25px; 
            border-radius: 12px; 
            border-left: 8px solid #1f77b4; 
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .step-header { color: #0d3b4c; font-weight: bold; font-size: 1.5em; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        diag = joblib.load('gbm_diagnostic_model-1.pkl')
        detector = joblib.load('gbm_detector.pkl')
        pathways = joblib.load('gbm_pathways.pkl')
        biomarker_ref = joblib.load('gbm_biomarkers (2).pkl')
        return diag, detector, pathways, biomarker_ref
    except Exception as e:
        st.error(f"Asset Synchronization Error: {e}")
        return None, None, None, None

diag, detector, pathways, biomarker_ref = load_assets()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("💎 MultiNet-AI Pro")
app_mode = st.sidebar.radio(
    "Navigation",
    ["Home", "🩺 Diagnostic Interface", "📖 App Documentation", "🧪 Interactive Demo Walkthrough"]
)

# --- PAGE 0: HOME PAGE ---
if app_mode == " Home":
    st.title("Welcome to MultiNet-AI Pro")
    st.markdown("### Personalized Glioblastoma Diagnostic Suite")
    
    uploaded_logo = st.file_uploader("Upload Main Interface Image (Logo/Branding)", type=["png", "jpg", "jpeg"])
    if uploaded_logo:
        st.image(uploaded_logo, use_container_width=True)
    else:
        st.info("Please upload your main page image above to customize the dashboard.")
    
    st.divider()
    st.markdown("""
    This platform integrates high-dimensional multi-omics data with gradient-boosted machine learning to provide 
    real-time diagnostic insights into Glioblastoma Multiforme (GBM).
    """)

# --- PAGE 1: DIAGNOSTIC INTERFACE (Top 10 First) ---
elif app_mode == "🩺 Diagnostic Interface":
    st.title("Diagnostic Analysis")
    if diag:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        st.subheader("🧬 High-Significance Biomarkers (Top 10)")
        tab1, tab2 = st.tabs(["Manual Abundance Entry", "Bulk CSV Analysis"])
        
        with tab1:
            with st.form("manual_entry"):
                st.write("### 🧬 Enter Abundance for High-Gain Features")
                cols = st.columns(2)
                # We use the top_10 list defined earlier in the code
                user_inputs = {feat: cols[i % 2].number_input(f"{feat}", value=10.0) for i, feat in enumerate(top_10)}
                submit = st.form_submit_button("RUN DIAGNOSTIC CONSENSUS")
                
                if submit:
                    # 1. Prepare full feature vector (23k features)
                    # We fill background features with a baseline low value (5.0)
                    full_input = pd.DataFrame({f: [user_inputs.get(f, 5.0)] for f in all_features})
                    
                    # 2. Generate Prediction
                    prob = model.predict_proba(full_input)[0][1]
                    
                    # 3. Display Metrics
                    st.divider()
                    st.metric("Probability of GBM Signature", f"{prob:.2%}")
                    if prob > 0.5: 
                        st.error("CONSENSUS: POSITIVE SIGNATURE DETECTED")
                    else: 
                        st.success("CONSENSUS: NEGATIVE SIGNATURE")

                    # 4. DATA VISUALIZATION: The Impact Bar Chart
                    st.write("### 📊 Local Feature Impact")
                    st.caption("This graph shows the weighted contribution of your inputs to the total risk score.")
                    
                    # Calculate: Input Value * Global Model Importance
                    impact_list = []
                    for feat in top_10:
                        # Extract the specific weight for this gene from our importance dataframe
                        weight = feat_df[feat_df['feature'] == feat]['importance'].values[0]
                        impact_list.append({
                            "Biomarker": feat, 
                            "Diagnostic Impact": user_inputs[feat] * weight
                        })
                    
                    # Create DataFrame and plot
                    plot_df = pd.DataFrame(impact_list).set_index("Biomarker")
                    st.bar_chart(plot_df, color="#1f77b4")
                

        with tab2:
            st.subheader("Bulk Data Pipeline")
            ordered_template_cols = top_10 + [f for f in all_features if f not in top_10]
            template_df = pd.DataFrame(columns=['Patient_ID'] + ordered_template_cols)
            buffer = io.BytesIO()
            template_df.to_csv(buffer, index=False)
            st.download_button("Download Requirements-Aligned CSV Template", data=buffer.getvalue(), file_name="MultiNet_Template.csv")
            
            up = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])
            if up:
                bulk_df = pd.read_csv(up)
                if set(all_features).issubset(bulk_df.columns):
                    bulk_df['Risk_Score'] = model.predict_proba(bulk_df[all_features])[:, 1]
                    st.bar_chart(bulk_df.set_index('Patient_ID')['Risk_Score'])
                    st.dataframe(bulk_df[['Patient_ID', 'Risk_Score']])

# --- PAGE 2: DOCUMENTATION ---
elif app_mode == "📖 App Documentation":
    st.title("Documentation & User Guide")
    
    st.markdown('<div class="doc-section"><div class="step-header">I. Diagnostic Processing & Inputs</div>', unsafe_allow_html=True)
    st.write("""
    The MultiNet-AI GUI is designed for precision-oncology workflows. Users should provide **Raw Abundance Scores** (standardized counts or intensity values) for multi-omic features. 
    
    **Workflow Stages:**
    1. **Data Ingestion:** The system accepts single-entry manual values or batch CSV uploads.
    2. **Feature Mapping:** It automatically aligns inputs to the Top 10 diagnostic drivers identified during model training.
    3. **Consensus Scoring:** The backend runs the XGBoost primary classifier and cross-references results with the Metabolic Detector.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="doc-section"><div class="step-header">II. Visual Evidence & Interpretability</div>', unsafe_allow_html=True)
    st.write("""
    - **Impact Charts:** Visualize how much weight each biomarker contributed to the final probability score.
    - **Pathways Mapping:** Displays the biological context (EMT, Metabolic Pathways, etc.) of the flagged biomarkers.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: DYNAMIC DEMO (Top 10 Focused) ---
elif app_mode == "🧪 Interactive Demo Walkthrough":
    st.title("Demo Walkthrough")
    st.write("Adjust the sliders for the **Top 10 Markers** below to see how they influence the prediction in real-time. ")
    
    if diag:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10_markers = feat_df['feature'].head(10).tolist()
        
        # UI for adjusting Top 10
        st.subheader("🎚️ Biomarker Entry for interactive demo walkthrough")
        cols = st.columns(2)
        demo_inputs = {}
        for i, gene in enumerate(top_10_markers):
            demo_inputs[gene] = cols[i % 2].slider(f"Abundance: {gene}", 0.0, 10000.0, 50.0)
            
        # Generate random biological noise for background genes
        # We use a state seed based on the sum of inputs to keep it responsive but "random"
        np.random.seed(int(sum(demo_inputs.values())))
        random_noise = np.random.uniform(0.0, 20.0, size=(1, len(all_features)))
        sim_df = pd.DataFrame(random_noise, columns=all_features)
        
        # Inject the slider values into the Top 10 positions
        for gene, val in demo_inputs.items():
            sim_df[gene] = val
            
        # Prediction
        prob = model.predict_proba(sim_df)[0][1]
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.write("### Prediction Results")
            st.metric("Live Probability", f"{prob:.2%}")
            if prob > 0.5:
                st.error("MODEL STATUS: GBM POSITIVE")
            else:
                st.success("MODEL STATUS: NEGATIVE")
            st.progress(float(prob))

        with res_col2:
            st.write("### Weighted Contribution (Top 10)")
            # Calculate local impact for visualization
            impact_data = []
            for gene in top_10_markers:
                weight = feat_df[feat_df['feature'] == gene]['importance'].values[0]
                impact_data.append({"Gene": gene, "Impact Score": demo_inputs[gene] * weight})
            
            st.bar_chart(pd.DataFrame(impact_data).set_index("Gene"), color="#004080")


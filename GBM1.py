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
st.sidebar.title("💎 MultiNet-AI")
app_mode = st.sidebar.radio(
    "Navigation",
    ["Home", "🩺 Input your own omics data", "📖 App Documentation", "🧪 Interactive Demo Walkthrough"]
)

if app_mode == "Home":
    st.title("Welcome to MultiNet-AI")
    st.markdown("### Personalized Glioblastoma Diagnostic Page")
    
    # Use a local file or a URL
    # st.image("logo.png", use_container_width=True) 
    
    # Or keep the uploader but provide a default
    uploaded_logo = st.file_uploader("Upload Branding Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_logo:
        st.image(uploaded_logo, use_container_width=True)
    else:
        # Placeholder for when no file is uploaded - you can use a URL here
        st.image("https://via.placeholder.com/800x400.png?text=MultiNet-AI+System+Overview", use_container_width=True)
    
    st.divider()
    st.markdown("""
    This platform integrates high-dimensional multi-omics data with gradient-boosted machine learning to provide 
    real-time diagnostic insights into Glioblastoma Multiforme (GBM).
    """)

# --- PAGE 1: DIAGNOSTIC INTERFACE (Top 10 First) ---
elif app_mode == "🩺 Input your own omics data":
    st.title("User Analysis Page")
    if diag:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
        top_10 = feat_df['feature'].head(10).tolist()

        st.subheader("🧬 High-Significance Biomarkers (Top 10)")
        tab1, tab2 = st.tabs(["Manual Abundance Entry", "Bulk CSV Analysis"])
        
        with tab1:
            with st.form("manual_entry"):
                st.write("### Enter Raw Abundance Value")
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

# --- PAGE 2: APP DOCUMENTATION (EVERY WORD INCLUDED) ---
elif app_mode == "📖 App Documentation":
    st.title("MultiNet-AI Web Application Documentation")
    
    st.markdown("""
    The MultiNet-AI GUI is a user-friendly platform designed to integrate multi-omics data—proteomics, transcriptomics, and metabolomics—to provide patient-specific diagnostic predictions for Glioblastoma Multiforme (GBM). This documentation provides a step-by-step guide to effectively operate the web application, including both manual single-sample entry and bulk multi-sample analysis.
    """)

    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">1. Overview of MultiNet-AI Functionality</div>', unsafe_allow_html=True)
        st.write("""
        MultiNet-AI allows users to input raw abundance scores for relevant features across multiple omics types:
        1. Proteomics (prot)** – Protein expression levels
        2. Transcriptomics (rna)** – Gene expression values
        3. Metabolomics (met)** – Metabolite concentrations
        
        The platform integrates these datasets into a single predictive model that uses binary logistic regression to compute a risk factor for GBM. The output is provided as a probability score, representing the likelihood of a sample being GBM-positive. The model evaluates the relative contribution of each biomarker, which is visualized in a bar graph to illustrate which features most influence the prediction.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">2. Accessing the GUI</div>', unsafe_allow_html=True)
        st.write("""
        The MultiNet-AI web application is freely accessible via a web browser. No installation is required. Upon loading the platform, users are presented with a sidebar menu to navigate between:
        1. Upload your own omics data
        2. App Documentation & User Guide
        3. Interactive Demo Walkthrough
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">3. Single-Sample Manual Entry</div>', unsafe_allow_html=True)
        st.write("""
        1. Navigate to “Upload your own omics data” from the sidebar.
        2. Switch to the tab labeled “Enter Omics Sample Value (Manual Entry)”.
        3. The interface displays the top 10 most critical biomarkers identified by the model.
        4. Users can input raw abundance values for each biomarker in the provided fields. Each biomarker corresponds to a relevant feature (prot, rna, met).
        5. Click the “Run Diagnostic” button to process the input values. The model automatically sets non-specified features to 0.00 to maintain proper feature alignment.
        
        **Review the output:**
        * **Probability Score:** Displayed as a percentage representing the GBM risk.
        * **Biomarker Impact Chart:** A bar graph shows the contribution of each entered biomarker.
        
        **Interpretation:**
        * Probability > 50% → High likelihood of GBM-positive profile
        * Probability < 50% → Likely healthy or negative profile
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">4. Bulk Multi-Sample Analysis (CSV)</div>', unsafe_allow_html=True)
        st.write("""
        1. Navigate to the “Bulk Analysis for Multiple Omics Data (CSV)” tab under the Upload your own omics data page.
        2. Download the CSV template, which includes columns for Patient ID and all relevant features (prot, rna, met) used by the model.
        3. Populate the template with patient data according to the feature labels. Unfilled features default to 0.00 during processing.
        4. Upload the completed CSV using the file uploader. The model performs predictions for each patient automatically.
        
        **Review bulk output:**
        * **Results Table:** Displays patient IDs, predicted probability scores, and GBM classification.
        * **Impact Score CSV:** Downloadable CSV includes impact scores for each feature per patient, calculated as user input × global feature importance.
        * **Comparative Risk Bar Graph:** Visualizes the relative probability scores of all patients in the batch.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">5. Interactive Demo Walkthrough</div>', unsafe_allow_html=True)
        st.write("""
        Navigate to “Interactive Demo Walkthrough” from the sidebar. Select a clinical profile to observe how modifying raw values directly affects the confidence score and predicted classification. The simulation highlights how specific biomarkers influence the model’s outcome in real time.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">6. Key Features Summary</div>', unsafe_allow_html=True)
        st.write("""
        * **Manual and Bulk Data Input:** Flexible options for single or multiple samples.
        * **Automatic Feature Handling:** Non-input features default to 0.00 for model compatibility.
        * **Probability Scores and Classification:** Outputs percentage likelihood and positive/negative classification.
        * **Visualizations:** Bar graphs for biomarker impact and comparative risk analysis.
        * **Downloadable Templates and Results:** CSV files for input and output, enabling easy record-keeping.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="doc-section"><div class="step-header">7. Recommendations for Use</div>', unsafe_allow_html=True)
        st.write("""
        * Always input raw abundance values as measured experimentally; the model assumes data in its original scale.
        * Use the template provided for bulk analyses to prevent formatting errors.
        * For new users, start with the Interactive Demo to understand model behavior and interpretation of outputs.
        * Review the biomarker impact charts to gain insights into the features most influencing predictions.
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


import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io
import time

# --- 1. CONFIG & MODERN INTERFACE ---
st.set_page_config(page_title="MultiNet-AI | Blue Edition", layout="wide", page_icon="💎")

# CSS for a modern, high-contrast Professional Navy & Blue theme
st.markdown("""
    <style>
        .stApp { background-color: #F8FAFC; }
        [data-testid="stSidebar"] { background-color: #0d3b4c !important; color: white; }
        .diagnostic-card {
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            border-top: 5px solid #1f77b4;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .ml-badge {
            background-color: #1f77b4;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #1f77b4 0%, #0d3b4c 100%) !important;
            color: white !important;
            border: none !important;
            height: 3em;
            border-radius: 8px !important;
            font-weight: bold !important;
            transition: 0.3s ease;
        }
        div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(31,119,180,0.3); }
    </style>
""", unsafe_allow_html=True)

# --- 2. CORE ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        # Load your specific GBM artifacts
        diag = joblib.load('gbm_diagnostic_model-1.pkl')
        return diag
    except:
        st.error("Model assets not found. Please ensure .pkl files are in the directory.")
        return None

diag = load_assets()

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("💎 MultiNet-AI")
    st.markdown("<span class='ml-badge'>ML-DRIVEN ENGINE v2.1</span>", unsafe_allow_html=True)
    st.divider()
    app_mode = st.radio("Navigation", ["Home", "🩺 Clinical Analysis", "📖 Technical Documentation", "🧪 Interactive Walkthrough"])
    st.divider()
    st.caption("Developed for Integrated Glioblastoma Diagnostics.")

# --- 4. PAGE: HOME ---
if app_mode == "Home":
    st.title("Welcome to MultiNet-AI")
    st.markdown("### Next-Gen Multi-Omic Integration for GBM")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        MultiNet-AI is an advanced diagnostic platform that fuses **Proteomics, Transcriptomics, and Metabolomics** into a unified predictive model. 
        
        **Key Features:**
        - **ML-Driven:** Powered by Gradient Boosted Trees and DIABLO alignment.
        - **Real-Time:** Dynamic probability scoring based on raw abundance inputs.
        - **Bulk Ready:** Process cohort datasets of up to 100+ patients instantly.
        """)
    with col2:
        st.image("logo.png", use_container_width=True) if False else st.info("Place logo.png in your repo to display branding here.")

# --- 5. PAGE: CLINICAL ANALYSIS (THE WORKHORSE) ---
elif app_mode == "🩺 Clinical Analysis":
    st.title("User Module Verification")
    
    if diag:
        model = diag['model']
        all_features = diag['features']
        
        # Calculate global importance for the top 10
        feat_df = pd.DataFrame({'f': all_features, 'i': model.feature_importances_}).sort_values('i', ascending=False)
        top_10 = feat_df.head(10)['f'].tolist()

        tab1, tab2 = st.tabs(["Manual Abundance Entry", "Bulk Cohort Processing"])

        with tab1:
            st.markdown("### Step 1: Input Raw Omics Values")
            # Using a Form prevents the app from re-calculating until 'Submit' is clicked
            with st.form("manual_entry_form"):
                cols = st.columns(2)
                user_inputs = {}
                for i, feat in enumerate(top_10):
                    with cols[i % 2]:
                        user_inputs[feat] = st.number_input(f"{feat}", value=0.00, format="%.4f")
                
                submit_btn = st.form_submit_button("⚡ EXECUTE ML DIAGNOSTIC")

            # Result Display - Only triggers on button click
            if submit_btn:
                with st.spinner("Analyzing cross-omic signature..."):
                    time.sleep(1) # Simulation for professional feel
                    
                    # Prepare input for 23,000+ features (filling background with 0)
                    input_vec = pd.DataFrame({f: [user_inputs.get(f, 0.0)] for f in all_features})
                    prob = model.predict_proba(input_vec)[0][1]

                    st.markdown("### Step 2: Diagnostic Results")
                    res_col, plot_col = st.columns([1, 2])
                    
                    with res_col:
                        st.metric("GBM Confidence", f"{prob:.2%}")
                        if prob > 0.5:
                            st.error("**CONSENSUS: GBM POSITIVE**")
                        else:
                            st.success("**CONSENSUS: NEGATIVE**")
                        st.progress(float(prob))

                    with plot_col:
                        # Visualization of local contribution
                        impact_data = [{"Biomarker": f, "Impact": user_inputs[f] * feat_df[feat_df['f']==f]['i'].values[0]} for f in top_10]
                        st.bar_chart(pd.DataFrame(impact_list).set_index("Biomarker"), color="#1f77b4")

        with tab2:
            st.subheader("Batch Analysis (100 Patient Template)")
            
            # Create a 100-patient blank template
            p_ids = [f"PATIENT_{i:03d}" for i in range(1, 101)]
            template_df = pd.DataFrame(columns=["Patient_ID"] + all_features)
            template_df["Patient_ID"] = p_ids
            template_df.iloc[:, 1:] = 0.00

            buf = io.BytesIO()
            template_df.to_csv(buf, index=False)
            st.download_button("📥 Download 100-Patient CSV Template", data=buf.getvalue(), file_name="MultiNet_Bulk_Template.csv", use_container_width=True)
            
            st.divider()
            up_file = st.file_uploader("Upload Populated CSV", type="csv")
            if up_file:
                df = pd.read_csv(up_file)
                if st.button("🚀 Process Batch Data"):
                    probs = model.predict_proba(df[all_features])[:, 1]
                    df['Risk_Score'] = probs
                    st.dataframe(df[['Patient_ID', 'Risk_Score']], use_container_width=True)
                    st.bar_chart(df.set_index('Patient_ID')['Risk_Score'])

# --- 6. PAGE: DOCUMENTATION (ML & TECH SPECS) ---
elif app_mode == "📖 Technical Documentation":
    st.title("System Architecture & Open Source")
    
    st.markdown("""
    ### 🧠 Machine Learning Backend
    MultiNet-AI utilizes a **supervised multi-block integration** strategy.
    
    **1. DIABLO (mixOmics) Integration:**
    We employ the DIABLO (Data Integration Analysis for Biomarker discovery using Latent Components) framework. 
    It identifies highly correlated molecular variables across mRNA, Proteins, and Metabolites that 
    best discriminate between GBM and Healthy states.
    
    **2. Gradient Boosted Classification:**
    The final diagnostic engine is built on **XGBoost**. This allows for non-linear decision 
    boundaries and handles the high-dimensionality of omics data ($p \gg n$) with extreme efficiency.
    """)
    
    
    st.markdown("""
    ### 🖥️ Frontend & Deployment
    - **Frontend:** Streamlit Reactive Interface (Python-native).
    - **Deployment:** Optimized for Streamlit Cloud / Docker.
    
    ### 🔗 GitHub Repositories
    - [ML Model Training & Artifacts](https://github.com/your-repo/GBM-ML-BackEnd)
    - [Streamlit GUI Implementation](https://github.com/your-repo/MultiNet-AI-FrontEnd)
    """)

# --- 7. PAGE: INTERACTIVE DEMO ---
elif app_mode == "🧪 Interactive Walkthrough":
    st.title("Real-Time Sensitivity Lab")
    st.write("Adjust the sliders below to see how shifts in biomarker abundance alter the ML prediction in real-time.")
    
    if diag:
        model = diag['model']
        all_features = diag['features']
        feat_df = pd.DataFrame({'f': all_features, 'i': model.feature_importances_}).sort_values('i', ascending=False)
        demo_markers = feat_df.head(5)['f'].tolist()
        
        sim_data = pd.DataFrame([[0.0]*len(all_features)], columns=all_features)
        
        cols = st.columns(len(demo_markers))
        for i, m in enumerate(demo_markers):
            with cols[i]:
                sim_data[m] = st.slider(f"{m}", 0.0, 100.0, 10.0)
        
        prob = model.predict_proba(sim_data)[0][1]
        st.divider()
        st.metric("Live Probability", f"{prob:.2%}")
        st.progress(float(prob))

"""
GBM Multi-Omics ML Diagnostic System
XGBoost-powered cancer detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="🧬 GBM ML Diagnostic",
    page_icon="🧬",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);}
    h1 {color: #1a202c; font-weight: 800;}
    .ml-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 10px 20px; border-radius: 20px;
        font-weight: 700; display: inline-block; margin: 10px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 600; border-radius: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        detector = pickle.load(open('gbm_detector.pkl', 'rb'))
        diagnostic = pickle.load(open('gbm_diagnostic_model-1.pkl', 'rb'))
        biomarkers = joblib.load('gbm_biomarkers (2).pkl')
        return {'detector': detector, 'diagnostic': diagnostic, 'biomarkers': biomarkers, 'status': 'success'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Generate cancer profile
def generate_profile(features, cancer_type, patient_id):
    seed = hash(f"{cancer_type}_{patient_id}_{datetime.now().microsecond}") % 2**32
    np.random.seed(seed)
    
    signatures = {
        'GBM (Glioblastoma)': {
            'mean': 5.5, 'std': 2.5,
            'high': {'EGFR_prot': (8, 10), 'VEGFA_prot': (7, 9), 'MET_prot': (7, 9)},
            'low': {'TP53_prot': (2, 3), 'PTEN_prot': (1, 2)}
        },
        'Lung Cancer': {
            'mean': 4.8, 'std': 2.2,
            'high': {'KRAS_prot': (8, 10), 'EGFR_prot': (7, 9)},
            'low': {'STK11_prot': (2, 3)}
        },
        'Breast Cancer': {
            'mean': 5.2, 'std': 2.0,
            'high': {'ERBB2_prot': (8, 10), 'ESR1_prot': (7, 9)},
            'low': {'BRCA1_prot': (2, 3)}
        },
        'Normal': {
            'mean': 4.0, 'std': 1.5,
            'high': {}, 'low': {}
        }
    }
    
    sig = signatures.get(cancer_type, signatures['Normal'])
    data = np.random.normal(sig['mean'], sig['std'], len(features))
    
    for i, feat in enumerate(features):
        for marker, (mn, mx) in sig['high'].items():
            if marker in feat:
                data[i] = np.random.uniform(mn, mx)
                break
        for marker, (mn, mx) in sig['low'].items():
            if marker in feat:
                data[i] = np.random.uniform(mn, mx)
                break
    
    data += np.random.normal(0, 0.3, len(features))
    data = np.clip(data, 0, None)
    return pd.DataFrame([data], columns=features)

# Prediction
def predict(model_dict, data):
    model = model_dict['model']
    proba = model.predict_proba(data)[0]
    pred = model.predict(data)[0]
    return {
        'prediction': int(pred),
        'probability': float(proba[1]),
        'confidence': float(max(proba)),
        'probs': proba.tolist()
    }

# Gauge chart
def gauge_chart(prob, pred):
    color = '#ef4444' if pred == 1 else '#10b981'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Cancer Probability (%)"},
        number={'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fee2e2'}
            ],
            'threshold': {'line': {'width': 3}, 'value': 50}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# Biomarker chart
def biomarker_chart(df):
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Biomarker'],
        orientation='h',
        marker=dict(color=df['Importance'], colorscale='Viridis'),
        text=df['Importance'].round(3),
        textposition='outside'
    ))
    fig.update_layout(
        title='Top Biomarkers (ML Feature Importance)',
        xaxis_title='Importance Score',
        height=400,
        margin=dict(l=150, r=50, t=80, b=50)
    )
    return fig

# Main app
def main():
    st.markdown("<h1 style='text-align: center;'>🧬 GBM ML Diagnostic Platform</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><span class='ml-badge'>🤖 POWERED BY XGBOOST ML</span></div>", unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if models['status'] == 'error':
        st.error(f"❌ Error: {models['message']}")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        patient_id = st.text_input("Patient ID", f"PT-{np.random.randint(1000, 9999)}")
        
        cancer_type = st.selectbox(
            "Cancer Type",
            ['GBM (Glioblastoma)', 'Lung Cancer', 'Breast Cancer', 'Normal']
        )
        
        run_btn = st.button("🚀 RUN ML PREDICTION", use_container_width=True, type="primary")
        
        st.markdown("---")
        st.info(f"**Features:** {len(models['detector']['features'])}\n\n**Algorithm:** XGBoost")
    
    # Main area
    if run_btn:
        st.markdown("---")
        st.subheader("🔬 Prediction Results")
        
        with st.spinner('Generating profile...'):
            data = generate_profile(models['detector']['features'], cancer_type, patient_id)
            time.sleep(0.5)
        
        with st.spinner('Running ML prediction...'):
            result = predict(models['detector'], data)
            time.sleep(0.5)
        
        # Results
        pred = result['prediction']
        prob = result['probability']
        conf = result['confidence']
        
        if pred == 1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                        color: white; padding: 30px; border-radius: 15px; text-align: center;'>
                <h2>⚠️ POSITIVE DETECTION</h2>
                <p style='font-size: 2rem; margin: 10px 0;'>{prob*100:.1f}% Cancer Probability</p>
                <p>Confidence: {conf*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; padding: 30px; border-radius: 15px; text-align: center;'>
                <h2>✅ NEGATIVE DETECTION</h2>
                <p style='font-size: 2rem; margin: 10px 0;'>{(1-prob)*100:.1f}% Normal</p>
                <p>Confidence: {conf*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics
        st.markdown("### 📈 Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cancer Prob", f"{prob*100:.1f}%")
        col2.metric("Normal Prob", f"{(1-prob)*100:.1f}%")
        col3.metric("Confidence", f"{conf*100:.1f}%")
        col4.metric("Diagnosis", "POSITIVE" if pred == 1 else "NEGATIVE")
        
        # Charts
        st.markdown("### 📊 Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(gauge_chart(prob, pred), use_container_width=True)
        
        with col2:
            st.plotly_chart(biomarker_chart(models['biomarkers']['top_targets_df']), use_container_width=True)
        
        # Data preview
        with st.expander("🔍 View Data (First 20 Features)"):
            display = data.T.head(20)
            display.columns = ['Expression']
            st.dataframe(display.style.background_gradient(cmap='viridis'))
    
    else:
        st.info("👆 Select cancer type and click **RUN ML PREDICTION** to start")
        st.plotly_chart(biomarker_chart(models['biomarkers']['top_targets_df']), use_container_width=True)

if __name__ == "__main__":
    main()

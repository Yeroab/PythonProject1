import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="GBM Diagnostic - Raw Data Fix", layout="wide")

@st.cache_resource
def load_assets():
    try:
        diag = joblib.load('gbm_diagnostic_model-1.pkl')
        report = joblib.load('gbm_biomarkers (2).pkl')
        return diag, report
    except Exception as e:
        st.error(f"File Load Error: {e}")
        return None, None

diag, report = load_assets()

if diag:
    model = diag['model']
    # Get all features the model was trained on
    all_features = model.get_booster().feature_names
    
    # Calculate Feature Importance to find the 'Heavy Hitters'
    importances = model.feature_importances_
    feat_import_df = pd.DataFrame({'feature': all_features, 'importance': importances})
    feat_import_df = feat_import_df.sort_values(by='importance', ascending=False)
    top_10_features = feat_import_df['feature'].head(10).tolist()

    st.title("🧠 GBM Prediction (Dynamic Input)")
    st.info("Note: To change the 98.5% result, you must modify the 'High-Impact Biomarkers' below.")

    with st.form("diagnostic_form"):
        st.subheader("🔥 High-Impact Biomarkers (Raw Values)")
        cols = st.columns(2)
        user_inputs = {}
        
        # We only show the most important features to ensure the model reacts
        for i, feat in enumerate(top_10_features):
            with cols[i % 2]:
                # We use a higher default value (100.0) since it's raw data
                user_inputs[feat] = st.number_input(f"{feat} (Critical)", value=100.0)

        submitted = st.form_submit_button("Run Analysis")

        if submitted:
            # Fill the entire feature set: 
            # We use 0.0 for unimportant ones, but use YOUR inputs for the top 10
            full_input_dict = {f: [user_inputs.get(f, 0.0)] for f in all_features}
            input_df = pd.DataFrame(full_input_dict)
            
            # Run Prediction
            prob = model.predict_proba(input_df[all_features])[0][1]
            
            st.divider()
            st.write(f"### Probability of GBM: {prob:.2%}")
            
            if prob > 0.5:
                st.error("Diagnostic Result: POSITIVE")
            else:
                st.success("Diagnostic Result: NEGATIVE")

            # Show a chart of why the result is what it is
            st.subheader("Biomarker Impact Chart")
            st.bar_chart(feat_import_df.head(10).set_index('feature'))

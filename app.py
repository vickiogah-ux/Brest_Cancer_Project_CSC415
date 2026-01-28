"""
Breast Cancer Prediction System - Web GUI
==========================================
Author: Ogah Victor (22CG031902)
Date: January 2026
Framework: Streamlit
Algorithm: Logistic Regression
Model Persistence: Joblib

This application provides a user-friendly interface to predict whether a tumor
is benign or malignant based on diagnostic features.

DISCLAIMER: This system is strictly for educational purposes and must not be 
presented as a medical diagnostic tool.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border-radius: 5px;
            height: 45px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0052a3;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================
@st.cache_resource
def load_model_and_scaler():
    """Load the pre-trained model and scaler from disk."""
    try:
        # Try to load from model directory
        model_path = Path(__file__).parent / "model" / "breast_cancer_model.pkl"
        scaler_path = Path(__file__).parent / "model" / "scaler.pkl"
        
        # Fallback paths if not found
        if not model_path.exists():
            model_path = Path("model/breast_cancer_model.pkl")
        if not scaler_path.exists():
            scaler_path = Path("model/scaler.pkl")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ============================================================================
# FEATURE INFORMATION
# ============================================================================
FEATURE_DESCRIPTIONS = {
    'radius_mean': 'Mean of distances from center to perimeter',
    'texture_mean': 'Mean of gray-scale values (texture)',
    'perimeter_mean': 'Mean size of the core tumor',
    'area_mean': 'Mean area of the tumor',
    'smoothness_mean': 'Mean of local variation in radius'
}

FEATURE_RANGES = {
    'radius_mean': (6.98, 28.11),
    'texture_mean': (9.71, 39.28),
    'perimeter_mean': (43.79, 188.5),
    'area_mean': (143.5, 2501.0),
    'smoothness_mean': (0.053, 0.163)
}

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Load model
    model, scaler = load_model_and_scaler()
    
    # Title and Header
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🏥 Breast Cancer Prediction System</h1>
            <p style='font-size: 16px; color: #666;'>
                Predict whether a tumor is <b>Benign</b> or <b>Malignant</b>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    with st.info():
        st.warning(
            "⚠️ **IMPORTANT DISCLAIMER**: This system is strictly for educational purposes. "
            "It should NOT be used as a substitute for professional medical diagnosis. "
            "Always consult with a qualified healthcare professional."
        )
    
    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    st.markdown("### 📊 Enter Tumor Diagnostic Features")
    st.markdown("*Provide the measurements below to get a prediction*")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        radius_mean = st.slider(
            "🔵 Radius Mean",
            min_value=FEATURE_RANGES['radius_mean'][0],
            max_value=FEATURE_RANGES['radius_mean'][1],
            value=12.0,
            step=0.1,
            help=FEATURE_DESCRIPTIONS['radius_mean']
        )
        
        texture_mean = st.slider(
            "🎨 Texture Mean",
            min_value=FEATURE_RANGES['texture_mean'][0],
            max_value=FEATURE_RANGES['texture_mean'][1],
            value=18.0,
            step=0.1,
            help=FEATURE_DESCRIPTIONS['texture_mean']
        )
        
        perimeter_mean = st.slider(
            "📏 Perimeter Mean",
            min_value=FEATURE_RANGES['perimeter_mean'][0],
            max_value=FEATURE_RANGES['perimeter_mean'][1],
            value=80.0,
            step=0.5,
            help=FEATURE_DESCRIPTIONS['perimeter_mean']
        )
    
    with col2:
        area_mean = st.slider(
            "📌 Area Mean",
            min_value=FEATURE_RANGES['area_mean'][0],
            max_value=FEATURE_RANGES['area_mean'][1],
            value=500.0,
            step=10.0,
            help=FEATURE_DESCRIPTIONS['area_mean']
        )
        
        smoothness_mean = st.slider(
            "🌊 Smoothness Mean",
            min_value=FEATURE_RANGES['smoothness_mean'][0],
            max_value=FEATURE_RANGES['smoothness_mean'][1],
            value=0.10,
            step=0.001,
            help=FEATURE_DESCRIPTIONS['smoothness_mean']
        )
    
    # ========================================================================
    # PREDICTION
    # ========================================================================
    if st.button("🔍 Make Prediction", key="predict_btn"):
        # Prepare input data
        input_features = np.array([[
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        # Diagnosis result
        if prediction == 0:
            diagnosis = "🔴 MALIGNANT"
            confidence = probability[0]
            color = "#FF6B6B"
            icon = "⚠️"
        else:
            diagnosis = "🟢 BENIGN"
            confidence = probability[1]
            color = "#51CF66"
            icon = "✓"
        
        # Display diagnosis with styling
        st.markdown(f"""
            <div style='
                background-color: {color}; 
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center;
                margin: 20px 0;
            '>
                <h2>{diagnosis}</h2>
                <p style='font-size: 18px; margin: 10px 0;'>
                    Confidence: <b>{confidence*100:.2f}%</b>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Detailed probability breakdown
        st.markdown("#### Probability Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "🔴 Malignant",
                f"{probability[0]*100:.2f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "🟢 Benign",
                f"{probability[1]*100:.2f}%",
                delta=None
            )
        
        # Input summary
        st.markdown("#### Input Features Summary")
        input_df = pd.DataFrame({
            'Feature': ['Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean'],
            'Value': [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]
        })
        st.table(input_df)
    
    # ========================================================================
    # SIDEBAR - INFORMATION
    # ========================================================================
    with st.sidebar:
        st.markdown("### 📖 About This Project")
        st.markdown("""
            **Algorithm:** Logistic Regression
            
            **Accuracy:** ~96-97%
            
            **Training Data:** Breast Cancer Wisconsin Dataset
            
            **Number of Features:** 5
            
            **Classes:** Benign / Malignant
        """)
        
        st.markdown("---")
        st.markdown("### 🎓 Model Information")
        st.markdown("""
            **Features Used:**
            - Radius Mean
            - Texture Mean
            - Perimeter Mean
            - Area Mean
            - Smoothness Mean
            
            **Preprocessing:**
            - StandardScaler normalization
            - 80-20 train-test split
            
            **Model Persistence:** Joblib
        """)
        
        st.markdown("---")
        st.markdown("### 👤 Author")
        st.markdown("""
            **Name:** Ogah Victor
            
            **Matric Number:** 22CG031902
            
            **Date:** January 2026
        """)
        
        st.markdown("---")
        st.markdown("""
            <p style='text-align: center; font-size: 12px; color: #999;'>
                Educational Purpose Only<br>
                Covenant University CSC 415
            </p>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

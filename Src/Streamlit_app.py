import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Dynamic Flight Pricing Engine",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pricing-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .price-increase {
        background-color: #ffcccc;
        border-left-color: #ff4444;
    }
    .price-decrease {
        background-color: #ccffcc;
        border-left-color: #44ff44;
    }
    .price-maintain {
        background-color: #f0f0f0;
        border-left-color: #888888;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class FlightPricingEngine:
    """Flight Dynamic Pricing Model Handler"""
    
    def __init__(self, model_path="pricing_model.pkl"):
        self.model_path = model_path
        self.model_artifacts = None
        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.label_encoders = None
        
    def load_model(self):
        """Load the trained pricing model and preprocessing artifacts"""
        try:
            self.model_artifacts = joblib.load(self.model_path)
            self.model = self.model_artifacts['model']
            self.feature_columns = self.model_artifacts['feature_columns']
            self.scaler = self.model_artifacts['scaler']
            self.label_encoders = self.model_artifacts['label_encoders']
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        try:
            processed_data = {}

            categorical_mappings = {
                'airline': ['Indigo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara', 'GoAir'],
                'source': ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad'],
                'destination': ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad'],
                'total_stops': ['non-stop', '1 stop', '2 stops', '3 stops'],
                'class_type': ['Economy', 'Business']
            }
            
            for col, categories in categorical_mappings.items():
                if col in input_data:
                    if input_data[col] in categories:
                        processed_data[col + '_encoded'] = categories.index(input_data[col])
                    else:
                        processed_data[col + '_encoded'] = 0

            numerical_features = {
                'duration_minutes': input_data.get('duration_minutes', 120),
                'dep_hour': input_data.get('dep_hour', 12),
                'arrival_hour': input_data.get('arrival_hour', 14),
                'is_peak_hour': input_data.get('is_peak_hour', 0),
                'days_left': input_data.get('days_left', 30),
                'demand_factor': input_data.get('demand_factor', 1.0),
                'competitor_price': input_data.get('competitor_price', input_data.get('current_price', 5000)),
                'is_weekend': input_data.get('is_weekend', 0),
                'booking_urgency': 1 / (input_data.get('days_left', 30) + 1),
                'route_popularity': input_data.get('route_popularity', 1000),
            }
            
            processed_data.update(numerical_features)

            current_price = input_data.get('current_price', 5000)
            competitor_price = input_data.get('competitor_price', current_price)
            
            processed_data['price_ratio_vs_competitor'] = current_price / competitor_price
            processed_data['price_advantage'] = (competitor_price - current_price) / competitor_price

            for feature in self.feature_columns:
                if feature not in processed_data:
                    processed_data[feature] = 0

            features_df = pd.DataFrame([processed_data])[self.feature_columns]
            
            return features_df
            
        except Exception as e:
            st.error(f"Error preprocessing data: {e}")
            return None
    
    def predict_optimal_price(self, input_data):
        """Predict optimal price with business constraints"""
        try:
            features_df = self.preprocess_input(input_data)
            if features_df is None:
                return None, None

            if self.model_artifacts['model_type'] == 'Linear Regression':
                features_scaled = self.scaler.transform(features_df)
                optimal_price = self.model.predict(features_scaled)[0]
            else:
                optimal_price = self.model.predict(features_df)[0]

            current_price = input_data.get('current_price', optimal_price)
            min_price = current_price * 0.7
            max_price = current_price * 1.5
            
            constrained_price = max(min_price, min(optimal_price, max_price))
            
            return constrained_price, optimal_price
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None
    
    def get_pricing_recommendation(self, current_price, optimal_price):
        """Get pricing recommendation based on optimal price"""
        price_change = ((optimal_price - current_price) / current_price) * 100
        
        if price_change > 10:
            return "INCREASE_PRICE_HIGH", price_change
        elif price_change > 5:
            return "INCREASE_PRICE_MEDIUM", price_change
        elif price_change < -10:
            return "DECREASE_PRICE_HIGH", price_change
        elif price_change < -5:
            return "DECREASE_PRICE_MEDIUM", price_change
        else:
            return "MAINTAIN_PRICE", price_change

def main():
    """Main Streamlit application"""

    st.markdown('<h1 class="main-header">✈️ Dynamic Flight Pricing Engine</h1>', 
                unsafe_allow_html=True)

    pricing_engine = FlightPricingEngine()

    with st.spinner("Loading pricing model..."):
        if not pricing_engine.load_model():
            st.error("Failed to load model. Please check if the model file exists.")
            return

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Pricing Mode",
        ["Single Flight Pricing", "Batch Pricing", "Revenue Analysis", "Model Info"]
    )
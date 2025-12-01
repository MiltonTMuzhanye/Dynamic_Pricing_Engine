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
    
    def __init__(self, model_path="../models/pricing_model.pkl"):
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

    if app_mode == "Single Flight Pricing":
        st.header("🔍 Single Flight Price Optimization")

        with st.form("flight_pricing_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                airline = st.selectbox("Airline", 
                                     ['Indigo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara', 'GoAir'])
                source = st.selectbox("Source City", 
                                    ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad'])
                destination = st.selectbox("Destination City", 
                                         ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad'])
                total_stops = st.selectbox("Number of Stops", 
                                         ['non-stop', '1 stop', '2 stops', '3 stops'])
                class_type = st.selectbox("Class Type", ['Economy', 'Business'])
                
            with col2:
                current_price = st.number_input("Current Price (₹)", 1000, 50000, 5000)
                competitor_price = st.number_input("Competitor Price (₹)", 1000, 50000, 4800)
                days_left = st.slider("Days Until Departure", 1, 90, 30)
                duration_minutes = st.slider("Flight Duration (minutes)", 60, 600, 120)
                dep_hour = st.slider("Departure Hour", 0, 23, 12)
                demand_factor = st.slider("Demand Factor", 0.3, 3.0, 1.0)
                is_weekend = st.checkbox("Weekend Flight")
                is_peak_hour = st.checkbox("Peak Hour Flight")
            
            submitted = st.form_submit_button("Optimize Price")

        if submitted:
            input_data = {
                'airline': airline,
                'source': source,
                'destination': destination,
                'total_stops': total_stops,
                'class_type': class_type,
                'current_price': current_price,
                'competitor_price': competitor_price,
                'days_left': days_left,
                'duration_minutes': duration_minutes,
                'dep_hour': dep_hour,
                'demand_factor': demand_factor,
                'is_weekend': 1 if is_weekend else 0,
                'is_peak_hour': 1 if is_peak_hour else 0,
                'route_popularity': 1000  # Simulated
            }

            with st.spinner("Calculating optimal price..."):
                constrained_price, raw_optimal_price = pricing_engine.predict_optimal_price(input_data)
            
            if constrained_price is not None:
                recommendation, price_change = pricing_engine.get_pricing_recommendation(
                    current_price, constrained_price
                )

                st.subheader("🎯 Pricing Optimization Results")

                if "INCREASE" in recommendation:
                    css_class = "price-increase"
                    emoji = "📈"
                elif "DECREASE" in recommendation:
                    css_class = "price-decrease" 
                    emoji = "📉"
                else:
                    css_class = "price-maintain"
                    emoji = "➡️"

                st.markdown(f"""
                <div class="pricing-box {css_class}">
                    <h3>{emoji} {recommendation.replace('_', ' ').title()}</h3>
                    <p><strong>Current Price:</strong> ₹{current_price:,.2f}</p>
                    <p><strong>Optimal Price:</strong> ₹{constrained_price:,.2f}</p>
                    <p><strong>Price Change:</strong> {price_change:+.2f}%</p>
                    <p><strong>Potential Revenue Impact:</strong> {abs(price_change):.2f}% per seat</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📋 Key Pricing Factors")
                
                factors = []
                if demand_factor > 1.5:
                    factors.append(f"High demand ({demand_factor:.1f}x normal)")
                if days_left < 7:
                    factors.append("Last-minute booking (high urgency)")
                if is_peak_hour:
                    factors.append("Peak hour flight premium")
                if is_weekend:
                    factors.append("Weekend travel premium")
                if competitor_price > current_price * 1.1:
                    factors.append("Competitor pricing allows increase")
                elif competitor_price < current_price * 0.9:
                    factors.append("Competitor undercutting price")
                
                if factors:
                    for factor in factors:
                        st.write(f"• {factor}")
                else:
                    st.write("• Market conditions suggest maintaining current price")

    elif app_mode == "Batch Pricing":
        st.header("📁 Batch Flight Pricing")
        
        st.info("""
        Upload a CSV file with flight data. Required columns:
        airline, source, destination, total_stops, class_type, current_price, 
        competitor_price, days_left, duration_minutes, dep_hour, demand_factor
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.write("Uploaded Flight Data Preview:")
                st.dataframe(batch_data.head())
                
                if st.button("Optimize Prices for All Flights"):
                    with st.spinner("Processing batch pricing optimization..."):
                        results = []
                        
                        for _, row in batch_data.iterrows():
                            try:
                                input_data = row.to_dict()
                                constrained_price, raw_optimal = pricing_engine.predict_optimal_price(input_data)
                                
                                if constrained_price is not None:
                                    recommendation, price_change = pricing_engine.get_pricing_recommendation(
                                        input_data['current_price'], constrained_price
                                    )
                                    
                                    results.append({
                                        'airline': input_data.get('airline', 'Unknown'),
                                        'route': f"{input_data.get('source', '')}-{input_data.get('destination', '')}",
                                        'current_price': input_data['current_price'],
                                        'optimal_price': constrained_price,
                                        'price_change_percent': price_change,
                                        'recommendation': recommendation,
                                        'days_left': input_data.get('days_left', 30)
                                    })
                            except Exception as e:
                                st.error(f"Error processing flight: {e}")
                        
                        results_df = pd.DataFrame(results)

                    st.subheader("📊 Batch Pricing Results")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Flights", len(results_df))
                    with col2:
                        increases = (results_df['price_change_percent'] > 5).sum()
                        st.metric("Price Increases", increases)
                    with col3:
                        decreases = (results_df['price_change_percent'] < -5).sum()
                        st.metric("Price Decreases", decreases)
                    with col4:
                        avg_change = results_df['price_change_percent'].mean()
                        st.metric("Avg Price Change", f"{avg_change:+.2f}%")

                    st.dataframe(results_df)

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Pricing Recommendations as CSV",
                        data=csv,
                        file_name=f"pricing_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")

    elif app_mode == "Revenue Analysis":
        st.header("💰 Revenue Impact Analysis")

        st.subheader("Potential Revenue Uplift")

        col1, col2 = st.columns(2)
        with col1:
            total_flights = st.number_input("Monthly Flight Volume", 100, 10000, 1000)
            avg_seats = st.number_input("Average Seats per Flight", 50, 300, 150)
            load_factor = st.slider("Load Factor (%)", 50, 100, 75) / 100
        with col2:
            current_avg_price = st.number_input("Current Average Price (₹)", 1000, 20000, 5000)
            expected_uplift = st.slider("Expected Price Uplift (%)", 0.0, 20.0, 8.5)
            implementation_rate = st.slider("Implementation Rate (%)", 0, 100, 80) / 100

        current_revenue = total_flights * avg_seats * load_factor * current_avg_price
        optimized_revenue = current_revenue * (1 + expected_uplift/100)
        potential_uplift = optimized_revenue - current_revenue
        achievable_uplift = potential_uplift * implementation_rate

        st.subheader("Financial Impact")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Monthly Revenue", f"₹{current_revenue:,.0f}")
        with col2:
            st.metric("Potential Revenue Uplift", f"₹{potential_uplift:,.0f}")
        with col3:
            st.metric("Achievable Uplift", f"₹{achievable_uplift:,.0f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Current Revenue', 'Potential Revenue', 'Achievable Revenue']
        values = [current_revenue, current_revenue + potential_uplift, current_revenue + achievable_uplift]
        
        bars = ax.bar(categories, values, color=['lightblue', 'lightgreen', 'salmon'])
        ax.set_ylabel('Revenue (₹)')
        ax.set_title('Revenue Impact of Dynamic Pricing')

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + current_revenue * 0.01,
                   f'₹{value:,.0f}', ha='center', va='bottom')
        
        st.pyplot(fig)

    else:
        st.header("ℹ️ Pricing Model Information")

        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", pricing_engine.model_artifacts['model_type'])
            st.metric("Number of Features", len(pricing_engine.feature_columns))
            
        with col2:
            metrics = pricing_engine.model_artifacts['performance_metrics']
            st.metric("RMSE", f"₹{metrics['rmse']:.2f}")
            st.metric("MAPE", f"{metrics['mape']:.2f}%")

        st.subheader("Business Impact")
        business_metrics = pricing_engine.model_artifacts['business_metrics']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Potential Revenue Uplift", f"{business_metrics['revenue_uplift']:.2f}%")
        with col2:
            st.metric("Average Price Change", f"{business_metrics['avg_price_change']:+.2f}%")

        st.subheader("Feature Importance")
        if hasattr(pricing_engine.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': pricing_engine.feature_columns,
                'importance': pricing_engine.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['feature'], feature_importance['importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Pricing Factors')
            ax.invert_yaxis()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dynamic Flight Pricing API",
    description="REST API for optimizing flight prices using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class FlightData(BaseModel):
    """Single flight data model"""
    flight_id: Optional[str] = Field(None, description="Unique flight identifier")
    airline: str = Field(..., description="Airline name")
    source: str = Field(..., description="Source city")
    destination: str = Field(..., description="Destination city")
    total_stops: str = Field(..., description="Number of stops", examples=["non-stop"])
    class_type: str = Field(..., description="Class type", examples=["Economy"])
    current_price: float = Field(..., ge=1000, le=50000, description="Current price in INR")
    competitor_price: float = Field(..., ge=1000, le=50000, description="Competitor price in INR")
    days_left: int = Field(..., ge=1, le=90, description="Days until departure")
    duration_minutes: int = Field(..., ge=60, le=600, description="Flight duration in minutes")
    dep_hour: int = Field(..., ge=0, le=23, description="Departure hour (0-23)")
    demand_factor: float = Field(..., ge=0.3, le=3.0, description="Demand multiplier")
    is_weekend: bool = Field(False, description="Is weekend flight")
    is_peak_hour: bool = Field(False, description="Is peak hour flight")

class PricingResponse(BaseModel):
    """Single pricing response model"""
    flight_id: Optional[str]
    current_price: float
    optimal_price: float
    price_change_percent: float
    recommendation: str
    timestamp: str
    factors: List[str]

class BatchPricingRequest(BaseModel):
    """Batch pricing request model"""
    flights: List[FlightData]

class BatchPricingResponse(BaseModel):
    """Batch pricing response model"""
    predictions: List[PricingResponse]
    summary: Dict[str, float]

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    model_loaded: bool
    model_type: Optional[str]
    timestamp: str

class PricingEngine:
    """Dynamic Pricing Engine for API"""
    
    def __init__(self, model_path: str = "../models/pricing_model.pkl"):
        self.model_path = model_path
        self.model_artifacts = None
        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.label_encoders = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained pricing model and preprocessing artifacts"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model_artifacts = joblib.load(self.model_path)
            self.model = self.model_artifacts['model']
            self.feature_columns = self.model_artifacts['feature_columns']
            self.scaler = self.model_artifacts['scaler']
            self.label_encoders = self.model_artifacts['label_encoders']
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_input(self, input_data: Dict) -> pd.DataFrame:
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
                'is_peak_hour': 1 if input_data.get('is_peak_hour', False) else 0,
                'days_left': input_data.get('days_left', 30),
                'demand_factor': input_data.get('demand_factor', 1.0),
                'competitor_price': input_data.get('competitor_price', input_data.get('current_price', 5000)),
                'is_weekend': 1 if input_data.get('is_weekend', False) else 0,
                'booking_urgency': 1 / (input_data.get('days_left', 30) + 1),
                'route_popularity': 1000,  # Simulated
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
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def predict_optimal_price(self, input_data: Dict) -> tuple:
        """Predict optimal price with business constraints"""
        try:
            features_df = self.preprocess_input(input_data)
            
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
            logger.error(f"Error making prediction: {e}")
            raise
    
    def get_pricing_recommendation(self, current_price: float, optimal_price: float) -> tuple:
        """Get pricing recommendation and factors"""
        price_change = ((optimal_price - current_price) / current_price) * 100
        
        if price_change > 10:
            recommendation = "INCREASE_PRICE_HIGH"
        elif price_change > 5:
            recommendation = "INCREASE_PRICE_MEDIUM"
        elif price_change < -10:
            recommendation = "DECREASE_PRICE_HIGH"
        elif price_change < -5:
            recommendation = "DECREASE_PRICE_MEDIUM"
        else:
            recommendation = "MAINTAIN_PRICE"
            
        return recommendation, price_change
    
    def analyze_pricing_factors(self, input_data: Dict, optimal_price: float) -> List[str]:
        """Analyze factors influencing pricing decision"""
        factors = []
        
        try:
            if input_data.get('demand_factor', 1.0) > 1.5:
                factors.append(f"High demand ({input_data['demand_factor']:.1f}x normal)")
            if input_data.get('days_left', 30) < 7:
                factors.append("Last-minute booking (high urgency)")
            if input_data.get('is_peak_hour', False):
                factors.append("Peak hour flight premium")
            if input_data.get('is_weekend', False):
                factors.append("Weekend travel premium")
            if input_data.get('competitor_price', 0) > input_data.get('current_price', 0) * 1.1:
                factors.append("Competitor pricing allows increase")
            elif input_data.get('competitor_price', 0) < input_data.get('current_price', 0) * 0.9:
                factors.append("Competitor undercutting price")
            if not factors:
                factors.append("Market conditions suggest current pricing is optimal")
                
        except Exception as e:
            logger.warning(f"Error analyzing pricing factors: {e}")
            factors.append("Factor analysis unavailable")
            
        return factors


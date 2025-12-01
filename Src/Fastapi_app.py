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
                'route_popularity': 1000,
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

pricing_engine = PricingEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting up Dynamic Pricing API")
    if not pricing_engine.load_model():
        logger.error("Failed to load model during startup")
        raise RuntimeError("Model loading failed")

@app.get("/", summary="API Root", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "message": "Dynamic Flight Pricing API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if pricing_engine.is_loaded else "unhealthy",
        model_loaded=pricing_engine.is_loaded,
        model_type=pricing_engine.model_artifacts['model_type'] if pricing_engine.is_loaded else None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if not pricing_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metrics = pricing_engine.model_artifacts['performance_metrics']
    business_metrics = pricing_engine.model_artifacts['business_metrics']
    
    return {
        "model_type": pricing_engine.model_artifacts['model_type'],
        "feature_count": len(pricing_engine.feature_columns),
        "performance_metrics": {
            "rmse": round(metrics['rmse'], 2),
            "mape": round(metrics['mape'], 2),
            "r2_score": round(metrics['r2'], 4)
        },
        "business_metrics": {
            "revenue_uplift": round(business_metrics['revenue_uplift'], 2),
            "avg_price_change": round(business_metrics['avg_price_change'], 2)
        },
        "loaded_at": "startup"
    }

@app.post("/pricing/predict", response_model=PricingResponse, tags=["Pricing"])
async def predict_pricing(flight_data: FlightData):
    """
    Predict optimal price for a single flight
    
    - **flight_data**: Flight information including current price, competitor data, and flight details
    - **returns**: Optimal pricing recommendation with analysis
    """
    if not pricing_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_dict = flight_data.dict()

        constrained_price, raw_optimal_price = pricing_engine.predict_optimal_price(input_dict)

        recommendation, price_change = pricing_engine.get_pricing_recommendation(
            flight_data.current_price, constrained_price
        )
        factors = pricing_engine.analyze_pricing_factors(input_dict, constrained_price)
        
        logger.info(f"Pricing prediction for flight {flight_data.flight_id}: "
                   f"current={flight_data.current_price}, optimal={constrained_price}, "
                   f"change={price_change:.2f}%")
        
        return PricingResponse(
            flight_id=flight_data.flight_id,
            current_price=flight_data.current_price,
            optimal_price=round(constrained_price, 2),
            price_change_percent=round(price_change, 2),
            recommendation=recommendation,
            timestamp=datetime.now().isoformat(),
            factors=factors
        )
        
    except Exception as e:
        logger.error(f"Pricing prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Pricing prediction failed: {str(e)}")

@app.post("/pricing/predict/batch", response_model=BatchPricingResponse, tags=["Pricing"])
async def predict_batch_pricing(batch_request: BatchPricingRequest):
    """
    Predict optimal prices for multiple flights in batch
    
    - **batch_request**: List of flight data objects
    - **returns**: Batch pricing predictions with summary statistics
    """
    if not pricing_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        total_flights = len(batch_request.flights)
        
        logger.info(f"Processing batch pricing for {total_flights} flights")
        
        for i, flight_data in enumerate(batch_request.flights):
            try:
                input_dict = flight_data.dict()
                constrained_price, raw_optimal_price = pricing_engine.predict_optimal_price(input_dict)
                recommendation, price_change = pricing_engine.get_pricing_recommendation(
                    flight_data.current_price, constrained_price
                )
                factors = pricing_engine.analyze_pricing_factors(input_dict, constrained_price)
                
                predictions.append(PricingResponse(
                    flight_id=flight_data.flight_id,
                    current_price=flight_data.current_price,
                    optimal_price=round(constrained_price, 2),
                    price_change_percent=round(price_change, 2),
                    recommendation=recommendation,
                    timestamp=datetime.now().isoformat(),
                    factors=factors
                ))
                
                if total_flights > 100 and (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{total_flights} flights")
                    
            except Exception as e:
                logger.error(f"Error processing flight {flight_data.flight_id}: {e}")

                predictions.append(PricingResponse(
                    flight_id=flight_data.flight_id,
                    current_price=flight_data.current_price,
                    optimal_price=flight_data.current_price,  # Fallback to current price
                    price_change_percent=0.0,
                    recommendation="ERROR",
                    timestamp=datetime.now().isoformat(),
                    factors=[f"Pricing error: {str(e)}"]
                ))

        successful_predictions = [p for p in predictions if p.recommendation != "ERROR"]
        if successful_predictions:
            avg_price_change = sum(p.price_change_percent for p in successful_predictions) / len(successful_predictions)
            price_increases = sum(1 for p in successful_predictions if p.price_change_percent > 5)
            price_decreases = sum(1 for p in successful_predictions if p.price_change_percent < -5)
            total_revenue_impact = sum(
                (p.optimal_price - p.current_price) for p in successful_predictions
            )
        else:
            avg_price_change = price_increases = price_decreases = total_revenue_impact = 0
        
        logger.info(f"Batch pricing completed: {len(successful_predictions)} successful, "
                   f"avg_change={avg_price_change:.2f}%")
        
        return BatchPricingResponse(
            predictions=predictions,
            summary={
                "total_flights": total_flights,
                "successful_predictions": len(successful_predictions),
                "failed_predictions": total_flights - len(successful_predictions),
                "average_price_change_percent": round(avg_price_change, 2),
                "flights_with_price_increases": price_increases,
                "flights_with_price_decreases": price_decreases,
                "total_revenue_impact": round(total_revenue_impact, 2)
            }
        )
        
    except Exception as e:
        logger.error(f"Batch pricing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch pricing failed: {str(e)}")

@app.post("/model/reload", tags=["Model"])
async def reload_model(background_tasks: BackgroundTasks):
    """
    Reload the model (admin endpoint)
    
    Useful when model files are updated without restarting the API
    """
    def reload_model_task():
        pricing_engine.load_model()
    
    background_tasks.add_task(reload_model_task)
    
    return {
        "message": "Model reload initiated",
        "timestamp": datetime.now().isoformat()
    }

@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all requests"""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"{request.method} {request.url.path} - "
               f"Status: {response.status_code} - "
               f"Time: {process_time:.3f}s")
    
    return response

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

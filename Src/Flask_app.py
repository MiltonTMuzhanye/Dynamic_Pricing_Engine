from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dynamic Flight Pricing</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .form-group { margin: 15px 0; }
        label { display: inline-block; width: 200px; font-weight: bold; }
        input, select { padding: 8px; width: 250px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
        .increase { background: #ffcccc; border-left: 5px solid #ff4444; }
        .decrease { background: #ccffcc; border-left: 5px solid #44ff44; }
        .maintain { background: #f0f0f0; border-left: 5px solid #888888; }
        .error { background: #ffcccc; border-left: 5px solid #ff0000; }
    </style>
</head>
<body>
    <div class="container">
        <h1>✈️ Dynamic Flight Pricing Engine</h1>
        
        <form method="POST" action="/web/predict">
            <div class="form-group">
                <label>Airline:</label>
                <select name="airline" required>
                    <option value="Indigo">Indigo</option>
                    <option value="Air India">Air India</option>
                    <option value="Jet Airways">Jet Airways</option>
                    <option value="SpiceJet">SpiceJet</option>
                    <option value="Vistara">Vistara</option>
                    <option value="GoAir">GoAir</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Source City:</label>
                <select name="source" required>
                    <option value="Delhi">Delhi</option>
                    <option value="Mumbai">Mumbai</option>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Kolkata">Kolkata</option>
                    <option value="Chennai">Chennai</option>
                    <option value="Hyderabad">Hyderabad</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Destination City:</label>
                <select name="destination" required>
                    <option value="Mumbai">Mumbai</option>
                    <option value="Delhi">Delhi</option>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Kolkata">Kolkata</option>
                    <option value="Chennai">Chennai</option>
                    <option value="Hyderabad">Hyderabad</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Current Price (₹):</label>
                <input type="number" name="current_price" value="5000" min="1000" max="50000" required>
            </div>
            
            <div class="form-group">
                <label>Competitor Price (₹):</label>
                <input type="number" name="competitor_price" value="4800" min="1000" max="50000" required>
            </div>
            
            <div class="form-group">
                <label>Days Until Departure:</label>
                <input type="number" name="days_left" value="30" min="1" max="90" required>
            </div>
            
            <div class="form-group">
                <label>Demand Factor:</label>
                <input type="number" name="demand_factor" value="1.0" min="0.3" max="3.0" step="0.1" required>
            </div>
            
            <div class="form-group">
                <label>Class Type:</label>
                <select name="class_type" required>
                    <option value="Economy">Economy</option>
                    <option value="Business">Business</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Stops:</label>
                <select name="total_stops" required>
                    <option value="non-stop">Non-stop</option>
                    <option value="1 stop">1 stop</option>
                    <option value="2 stops">2 stops</option>
                    <option value="3 stops">3 stops</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Duration (minutes):</label>
                <input type="number" name="duration_minutes" value="120" min="60" max="600" required>
            </div>
            
            <div class="form-group">
                <label>Departure Hour:</label>
                <input type="number" name="dep_hour" value="12" min="0" max="23" required>
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" name="is_weekend" value="true">
                    Weekend Flight
                </label>
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" name="is_peak_hour" value="true">
                    Peak Hour Flight
                </label>
            </div>
            
            <button type="submit">Optimize Price</button>
        </form>
        
        {% if result %}
        <div class="result {{ result.css_class }}">
            <h3>{{ result.emoji }} {{ result.recommendation }}</h3>
            <p><strong>Current Price:</strong> ₹{{ result.current_price }}</p>
            <p><strong>Optimal Price:</strong> ₹{{ result.optimal_price }}</p>
            <p><strong>Price Change:</strong> {{ result.price_change }}</p>
            <p><strong>Factors:</strong></p>
            <ul>
                {% for factor in result.factors %}
                <li>{{ factor }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

class FlaskPricingEngine:
    """Dynamic Pricing Engine for Flask"""
    
    def __init__(self, model_path: str = "../models/pricing_model.pkl"):
        self.model_path = model_path
        self.model_artifacts = None
        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.label_encoders = None
        
    def load_model(self):
        """Load the trained pricing model and preprocessing artifacts"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model_artifacts = joblib.load(self.model_path)
            self.model = self.model_artifacts['model']
            self.feature_columns = self.model_artifacts['feature_columns']
            self.scaler = self.model_artifacts['scaler']
            self.label_encoders = self.model_artifacts['label_encoders']
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        try:
            processed_data = {}

            numeric_fields = ['current_price', 'competitor_price', 'days_left', 
                            'duration_minutes', 'dep_hour', 'demand_factor']
            for field in numeric_fields:
                if field in input_data:
                    processed_data[field] = float(input_data[field])

            processed_data['is_weekend'] = 1 if input_data.get('is_weekend') == 'true' else 0
            processed_data['is_peak_hour'] = 1 if input_data.get('is_peak_hour') == 'true' else 0

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

            processed_data['booking_urgency'] = 1 / (processed_data.get('days_left', 30) + 1)
            processed_data['route_popularity'] = 1000  # Simulated

            current_price = processed_data.get('current_price', 5000)
            competitor_price = processed_data.get('competitor_price', current_price)
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
    
    def predict_optimal_price(self, input_data: dict) -> tuple:
        """Predict optimal price with business constraints"""
        try:
            features_df = self.preprocess_input(input_data)
            
            if self.model_artifacts['model_type'] == 'Linear Regression':
                features_scaled = self.scaler.transform(features_df)
                optimal_price = self.model.predict(features_scaled)[0]
            else:
                optimal_price = self.model.predict(features_df)[0]

            current_price = input_data.get('current_price', optimal_price)
            if isinstance(current_price, str):
                current_price = float(current_price)
                
            min_price = current_price * 0.7
            max_price = current_price * 1.5
            
            constrained_price = max(min_price, min(optimal_price, max_price))
            
            return constrained_price, optimal_price
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def get_pricing_recommendation(self, current_price: float, optimal_price: float) -> dict:
        """Get pricing recommendation for web display"""
        price_change = ((optimal_price - current_price) / current_price) * 100
        
        if price_change > 10:
            recommendation = "SIGNIFICANT PRICE INCREASE RECOMMENDED"
            css_class = "increase"
            emoji = "📈"
        elif price_change > 5:
            recommendation = "MODERATE PRICE INCREASE RECOMMENDED"
            css_class = "increase"
            emoji = "📈"
        elif price_change < -10:
            recommendation = "SIGNIFICANT PRICE DECREASE RECOMMENDED"
            css_class = "decrease"
            emoji = "📉"
        elif price_change < -5:
            recommendation = "MODERATE PRICE DECREASE RECOMMENDED"
            css_class = "decrease"
            emoji = "📉"
        else:
            recommendation = "MAINTAIN CURRENT PRICE"
            css_class = "maintain"
            emoji = "➡️"
            
        return {
            'recommendation': recommendation,
            'css_class': css_class,
            'emoji': emoji,
            'price_change': f"{price_change:+.2f}%"
        }
    
    def analyze_pricing_factors(self, input_data: dict) -> list:
        """Analyze factors influencing pricing decision"""
        factors = []
        
        try:
            demand_factor = float(input_data.get('demand_factor', 1.0))
            if demand_factor > 1.5:
                factors.append(f"High demand ({demand_factor:.1f}x normal)")
                
            days_left = int(input_data.get('days_left', 30))
            if days_left < 7:
                factors.append("Last-minute booking (high urgency)")
                
            if input_data.get('is_peak_hour') == 'true':
                factors.append("Peak hour flight premium")
                
            if input_data.get('is_weekend') == 'true':
                factors.append("Weekend travel premium")
                
            current_price = float(input_data.get('current_price', 5000))
            competitor_price = float(input_data.get('competitor_price', current_price))
            
            if competitor_price > current_price * 1.1:
                factors.append("Competitor pricing allows increase")
            elif competitor_price < current_price * 0.9:
                factors.append("Competitor undercutting price")
                
            if not factors:
                factors.append("Market conditions suggest current pricing is optimal")
                
        except Exception as e:
            logger.warning(f"Error analyzing pricing factors: {e}")
            factors.append("Factor analysis unavailable")
            
        return factors

pricing_engine = FlaskPricingEngine()

@app.before_first_request
def load_model():
    """Load model before first request"""
    if not pricing_engine.load_model():
        logger.error("Failed to load model")
        raise RuntimeError("Model loading failed")

@app.route('/')
def home():
    """Home page with pricing form"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/web/predict', methods=['POST'])
def web_predict():
    """Web form prediction endpoint"""
    try:

        form_data = request.form.to_dict()

        constrained_price, raw_optimal_price = pricing_engine.predict_optimal_price(form_data)

        current_price = float(form_data['current_price'])
        recommendation_data = pricing_engine.get_pricing_recommendation(current_price, constrained_price)
        factors = pricing_engine.analyze_pricing_factors(form_data)
        
        result = {
            'current_price': f"{current_price:,.2f}",
            'optimal_price': f"{constrained_price:,.2f}",
            'factors': factors,
            **recommendation_data
        }
        
        logger.info(f"Web prediction: {current_price} -> {constrained_price}")
        
        return render_template_string(HTML_TEMPLATE, result=result)
        
    except Exception as e:
        logger.error(f"Web prediction error: {e}")
        error_result = {
            'recommendation': 'PRICING ERROR',
            'css_class': 'error',
            'emoji': '❌',
            'current_price': 'N/A',
            'optimal_price': 'N/A',
            'price_change': 'N/A',
            'factors': [f'Error: {str(e)}']
        }
        return render_template_string(HTML_TEMPLATE, result=error_result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API for single prediction"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()

        required_fields = ['airline', 'source', 'destination', 'current_price', 'competitor_price']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        constrained_price, raw_optimal_price = pricing_engine.predict_optimal_price(data)
        recommendation_data = pricing_engine.get_pricing_recommendation(
            data['current_price'], constrained_price
        )
        factors = pricing_engine.analyze_pricing_factors(data)
        
        response = {
            'flight_id': data.get('flight_id'),
            'current_price': data['current_price'],
            'optimal_price': round(constrained_price, 2),
            'price_change_percent': float(recommendation_data['price_change'].replace('%', '')),
            'recommendation': recommendation_data['recommendation'],
            'factors': factors,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"API prediction: {data['current_price']} -> {constrained_price}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict/batch', methods=['POST'])
def api_predict_batch():
    """JSON API for batch predictions"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        if 'flights' not in data or not isinstance(data['flights'], list):
            return jsonify({'error': 'Missing or invalid flights list'}), 400
        
        predictions = []
        successful = 0
        failed = 0
        
        for i, flight_data in enumerate(data['flights']):
            try:
                constrained_price, raw_optimal_price = pricing_engine.predict_optimal_price(flight_data)
                recommendation_data = pricing_engine.get_pricing_recommendation(
                    flight_data['current_price'], constrained_price
                )
                factors = pricing_engine.analyze_pricing_factors(flight_data)
                
                predictions.append({
                    'flight_id': flight_data.get('flight_id'),
                    'current_price': flight_data['current_price'],
                    'optimal_price': round(constrained_price, 2),
                    'price_change_percent': float(recommendation_data['price_change'].replace('%', '')),
                    'recommendation': recommendation_data['recommendation'],
                    'factors': factors,
                    'timestamp': datetime.now().isoformat()
                })
                successful += 1
                
            except Exception as e:
                logger.error(f"Error processing flight {i}: {e}")
                predictions.append({
                    'flight_id': flight_data.get('flight_id'),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                failed += 1

        successful_predictions = [p for p in predictions if 'error' not in p]
        if successful_predictions:
            avg_price_change = sum(p['price_change_percent'] for p in successful_predictions) / len(successful_predictions)
            total_revenue_impact = sum(
                p['optimal_price'] - p['current_price'] for p in successful_predictions
            )
        else:
            avg_price_change = total_revenue_impact = 0
        
        response = {
            'predictions': predictions,
            'summary': {
                'total_flights': len(data['flights']),
                'successful_predictions': successful,
                'failed_predictions': failed,
                'average_price_change_percent': round(avg_price_change, 2),
                'total_revenue_impact': round(total_revenue_impact, 2)
            }
        }
        
        logger.info(f"Batch API prediction completed: {successful} successful, {failed} failed")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch API prediction error: {e}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy',
        'model_loaded': pricing_engine.model is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    if pricing_engine.model_artifacts:
        health_status['model_type'] = pricing_engine.model_artifacts['model_type']
        health_status['feature_count'] = len(pricing_engine.feature_columns)
    
    return jsonify(health_status)

@app.route('/api/model/info', methods=['GET'])
def api_model_info():
    """Get model information"""
    if not pricing_engine.model_artifacts:
        return jsonify({'error': 'Model not loaded'}), 503
    
    metrics = pricing_engine.model_artifacts['performance_metrics']
    business_metrics = pricing_engine.model_artifacts['business_metrics']
    
    info = {
        'model_type': pricing_engine.model_artifacts['model_type'],
        'feature_count': len(pricing_engine.feature_columns),
        'performance_metrics': {
            'rmse': round(metrics['rmse'], 2),
            'mape': round(metrics['mape'], 2),
            'r2_score': round(metrics['r2'], 4)
        },
        'business_metrics': {
            'revenue_uplift': round(business_metrics['revenue_uplift'], 2),
            'avg_price_change': round(business_metrics['avg_price_change'], 2)
        }
    }
    
    return jsonify(info)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if not pricing_engine.load_model():
        logger.error("Failed to load model. Exiting.")
        exit(1)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
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


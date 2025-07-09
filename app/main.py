# app/main.py
import os
import sys
import pandas as pd
from flask import Flask, request, render_template, jsonify

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import make_prediction
from src.config import PROCESSED_MASTER_PATH

# --- Pre-load data for localities when the app starts ---
try:
    df = pd.read_csv(PROCESSED_MASTER_PATH, usecols=['City', 'Locality'])
    df.dropna(subset=['City', 'Locality'], inplace=True)
    # Create a dictionary for fast lookup: { "CityName": ["Locality1", "Locality2", ...], ... }
    LOCALITY_DATA = df.groupby('City')['Locality'].apply(lambda x: sorted(list(x.unique()))).to_dict()
    print("Successfully loaded and prepared locality data.")
except FileNotFoundError:
    print(f"WARNING: Could not find master data at {PROCESSED_MASTER_PATH}. Locality suggestions will not work.")
    LOCALITY_DATA = {}
# ---------------------------------------------------------

app = Flask(__name__)

# --- Web Routes ---

@app.route('/', methods=['GET'])
def index():
    """
    Renders the main HTML page.
    Passes the list of cities and initial localities for the default city.
    """
    cities = sorted(list(LOCALITY_DATA.keys()))
    # Get localities for the first city in the list as default
    initial_localities = LOCALITY_DATA.get(cities[0], []) if cities else []
    
    return render_template('index.html', cities=cities, localities=initial_localities)

@app.route('/get_localities/<city_name>', methods=['GET'])
def get_localities(city_name):
    """
    API endpoint to get a list of localities for a given city.
    """
    localities = LOCALITY_DATA.get(city_name, [])
    return jsonify({'localities': localities})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives property data from the form, passes it to the ML model
    for a prediction, and returns the result.
    """
    try:
        form_data = request.form.to_dict()
        print(f"Received form data: {form_data}")
        input_df = pd.DataFrame([form_data])
        
        prediction = make_prediction(input_df)
        
        response = {
            'success': True,
            'estimated_price': f'₹ {prediction:,.0f}'
        }
        return jsonify(response)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        error_response = {
            'success': False,
            'error': str(e)
        }
        return jsonify(error_response), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
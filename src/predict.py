# src/predict.py
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any

# Import the necessary functions and config from our project
from src import config, utils

logger = utils.get_logger(__name__)

# --- Re-use the same feature engineering functions from train.py ---
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> np.ndarray:
    """Calculate the distance between two points on Earth in kilometers."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds distance from city center as a new feature."""
    logger.info("..._predict: Creating geospatial features for new data...")
    city_centers = {
        'Mumbai': (19.0760, 72.8777), 'Gurugram': (28.4595, 77.0266),
        'Noida': (28.5355, 77.3910), 'Hyderabad': (17.3850, 78.4867),
        'Kolkata': (22.5726, 88.3639), 'Thane': (19.2183, 72.9781),
        'Pune': (18.5204, 73.8567), 'Bangalore': (12.9716, 77.5946),
        'Chennai': (13.0827, 80.2707), 'Ahmedabad': (23.0225, 72.5714)
    }
    
    city_map = df['City'].map(city_centers)
    center_lats, center_lons = zip(*city_map.apply(lambda x: x if isinstance(x, tuple) else (np.nan, np.nan)))
    
    df['distance_to_center'] = haversine_distance(
        df['Latitude'], df['Longitude'],
        pd.Series(center_lats, index=df.index), pd.Series(center_lons, index=df.index)
    )
    return df
# ----------------------------------------------------------------

class PredictionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing Prediction Service...")
            cls._instance = super(PredictionService, cls).__new__(cls)
            try:
                cls.pipeline = joblib.load(config.PIPELINE_PATH)
                logger.info("Prediction pipeline loaded successfully.")
            except FileNotFoundError as e:
                logger.error(f"Failed to load model artifacts: {e}. Run the training pipeline first.")
                cls.pipeline = None
        return cls._instance

    def predict(self, input_data: pd.DataFrame) -> float:
        if self.pipeline is None:
            raise RuntimeError("Model artifacts are not loaded.")

        # --- THIS IS THE FIX ---
        # 1. Convert columns to their base types first
        input_df = input_data.copy()
        for col in config.NUMERICAL_FEATURES:
            if col != 'distance_to_center': # This column doesn't exist yet
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # 2. Apply the SAME feature engineering steps as in training
        input_df_with_features = add_geospatial_features(input_df)
        
        # 3. Ensure all columns the model expects are present
        for col in config.MODEL_FEATURES:
            if col not in input_df_with_features.columns:
                input_df_with_features[col] = np.nan
        
        # 4. Select columns in the correct order
        final_input_df = input_df_with_features[config.MODEL_FEATURES]
        # -----------------------

        logger.info("Applying preprocessing pipeline and making prediction...")
        prediction_log = self.pipeline.predict(final_input_df)
        
        prediction_original = np.expm1(prediction_log)
        
        return float(prediction_original[0])

prediction_service = PredictionService()

def make_prediction(input_data: pd.DataFrame) -> float:
    return prediction_service.predict(input_data)
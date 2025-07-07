# src/predict.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from typing import Dict, Any
from src import config, utils

logger = utils.get_logger(__name__)

class PredictionService:
    """A singleton class to load and hold the model and pipeline."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing Prediction Service...")
            cls._instance = super(PredictionService, cls).__new__(cls)
            try:
                cls.pipeline = joblib.load(config.PIPELINE_PATH)
                cls.model = xgb.XGBRegressor()
                cls.model.load_model(config.MODEL_PATH)
                logger.info("Prediction pipeline and model loaded successfully.")
            except FileNotFoundError as e:
                logger.error(f"Failed to load model artifacts: {e}. Run the training pipeline first.")
                cls.pipeline = None
                cls.model = None
        return cls._instance

    def predict(self, input_data: Dict[str, Any]) -> float:
        """
        Makes a price prediction on a single instance of input data.
        """
        if self.pipeline is None or self.model is None:
            raise RuntimeError("Model artifacts are not loaded.")

        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in config.MODEL_FEATURES:
            if col not in input_df.columns:
                input_df[col] = np.nan

        # Select columns in the correct order
        input_df = input_df[config.MODEL_FEATURES]
        
        # Preprocess the input data
        processed_input = self.pipeline.transform(input_df)
        
        # Make prediction
        prediction_log = self.model.predict(processed_input)
        
        # Convert back to original scale and return
        prediction_original = np.expm1(prediction_log)
        
        return float(prediction_original[0])

# Instantiate the service once when the module is loaded
prediction_service = PredictionService()

def make_prediction(input_data: Dict[str, Any]) -> float:
    """Public function to access the prediction service."""
    return prediction_service.predict(input_data)
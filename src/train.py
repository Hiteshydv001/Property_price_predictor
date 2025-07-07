# src/train.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
from src import config, utils, feature_engineering

# Initialize the logger for this module
logger = utils.get_logger(__name__)

def run_training():
    """
    Main orchestrator function for the training pipeline.
    This loads data, engineers features, trains the model, and saves all artifacts.
    """
    logger.info("Starting model training process...")
    
    # 1. Load the processed master dataset
    try:
        df = pd.read_csv(config.PROCESSED_MASTER_PATH, low_memory=False)
        logger.info(f"Successfully loaded processed data with {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"FATAL: Processed data not found at {config.PROCESSED_MASTER_PATH}. Please run the data processing step first.")
        return

    # 2. Engineer features using the dedicated feature engineering pipeline
    X, y_log = feature_engineering.run_feature_engineering(df)
    
    # 3. Split the fully processed data into training and validation sets
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    logger.info(f"Data split complete: Training on {X_train.shape[0]} samples, validating on {X_test.shape[0]} samples.")
    
    # 4. Initialize and train the XGBoost model
    logger.info("Training XGBoost model...")
    
    # Convert data to DMatrix format for better performance
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dtest = xgb.DMatrix(X_test, label=y_test_log)
    
    # Train using native API
    model = xgb.train(
        params=config.XGBOOST_PARAMS,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtest, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    logger.info("Model training complete. Evaluating performance on the validation set...")
    
    # 5. Evaluate the model's performance
    predictions_log = model.predict(dtest)
    predictions_original = np.expm1(predictions_log)
    y_test_original = np.expm1(y_test_log)
    
    r2 = r2_score(y_test_original, predictions_original)
    mae = mean_absolute_error(y_test_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    
    logger.info("--- Model Performance ---")
    logger.info(f"  R-squared (R²): {r2:.4f}")
    logger.info(f"  Mean Absolute Error (MAE): ₹{mae:,.0f}")
    logger.info(f"  Root Mean Squared Error (RMSE): ₹{rmse:,.0f}")

    # 6. Save the final trained model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, 'property_price_xgboost.json')
    model.save_model(model_path)
    logger.info(f"Model saved successfully to {model_path}")
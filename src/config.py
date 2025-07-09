# src/config.py
import os

# --- Project Directory Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- File Paths ---
PROCESSED_MASTER_PATH = os.path.join(PROCESSED_DATA_DIR, 'master_india_properties.csv')
PIPELINE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'property_price_xgboost.json')

# Raw file paths
PROPERTIES_FILE = os.path.join(RAW_DATA_DIR, 'properties.csv')
MAKAAN_FILE = os.path.join(RAW_DATA_DIR, 'Makaan_Properties_Buy.csv')
ACRES_FOLDER = os.path.join(RAW_DATA_DIR, '99acre')
ACRES_FACETS_PATH = os.path.join(ACRES_FOLDER, 'facets', 'facets')
ACRES_CITY_FILES = {
    'Gurugram': os.path.join(ACRES_FOLDER, 'gurgaon_10k.csv'),
    'Hyderabad': os.path.join(ACRES_FOLDER, 'hyderabad.csv'),
    'Kolkata': os.path.join(ACRES_FOLDER, 'kolkata.csv'),
    'Mumbai': os.path.join(ACRES_FOLDER, 'mumbai.csv')
}

# --- Feature & Model Configuration ---
TARGET_COLUMN = 'Price'
FEATURES_TO_DROP = ['Bathrooms', 'Floor', 'Property_Age']

# --- THIS IS THE CHANGE ---
# 'Bedrooms' has been renamed to 'Rooms' for better generalization
NUMERICAL_FEATURES = ['Area_SqFt', 'Rooms', 'Longitude', 'Latitude', 'Total_Floors', 'distance_to_center']
# ---------------------------

CATEGORICAL_FEATURES = ['City', 'Property_Type', 'Furnishing_Status', 'Area_Type', 'Source_Dataset']
HIGH_CARDINALITY_FEATURES = ['Locality']
AMENITY_FEATURES = ['has_Pool', 'has_Gym', 'has_Lift', 'has_Parking']

# Combine all features to be used by the model
MODEL_FEATURES = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES + 
    HIGH_CARDINALITY_FEATURES + AMENITY_FEATURES
)

# --- XGBoost/LightGBM Model Hyperparameters ---
# This can be used for any model that accepts these parameters
MODEL_PARAMS = {
    'objective': 'regression_l1', # MAE is less sensitive to outliers than MSE
    'metric': 'rmse',
    'n_estimators': 1500,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1, # L1 Regularization
    'lambda_l2': 0.1, # L2 Regularization
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}
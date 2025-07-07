# src/config.py
import os

# --- Path Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw Data Paths
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROPERTIES_FILE = os.path.join(RAW_DATA_DIR, 'properties.csv')
MAKAAN_FILE = os.path.join(RAW_DATA_DIR, 'Makaan_Properties_Buy.csv')

# 99acres Data
ACRES_DATA_DIR = os.path.join(RAW_DATA_DIR, '99acre')
ACRES_FACETS_PATH = os.path.join(ACRES_DATA_DIR, 'facets')
ACRES_CITY_FILES = {
    'Mumbai': os.path.join(ACRES_DATA_DIR, 'mumbai.csv'),
    'Hyderabad': os.path.join(ACRES_DATA_DIR, 'hyderabad.csv'),
    'Kolkata': os.path.join(ACRES_DATA_DIR, 'kolkata.csv'),
    'Gurgaon': os.path.join(ACRES_DATA_DIR, 'gurgaon_10k.csv')
}

# Processed Data Paths
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
PROCESSED_MASTER_PATH = os.path.join(PROCESSED_DATA_DIR, 'master_india_properties.csv')

# Model Paths
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# --- Feature Configuration ---
TARGET_COLUMN = 'Price'

FEATURES_TO_DROP = [
    'Source_Dataset',  # Not relevant for prediction
    'Latitude',        # Too sparse in current data
    'Longitude'       # Too sparse in current data
]

# Feature Groupings for Pipeline
NUMERICAL_FEATURES = [
    'Area_SqFt',
    'Bedrooms',
    'Bathrooms',
    'Floor',
    'Total_Floors'
]

CATEGORICAL_FEATURES = [
    'City',
    'Property_Type',
    'Furnishing_Status',
    'Property_Age',
    'Area_Type'
]

HIGH_CARDINALITY_FEATURES = [
    'Locality'
]

BINARY_FEATURES = [
    'has_Pool',
    'has_Gym',
    'has_Lift',
    'has_Parking'
]

# Combined Features List
MODEL_FEATURES = (
    NUMERICAL_FEATURES +
    CATEGORICAL_FEATURES +
    HIGH_CARDINALITY_FEATURES +
    BINARY_FEATURES
)

# --- Model Configuration ---
RANDOM_SEED = 42
N_FOLDS = 5

# XGBoost Parameters
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': RANDOM_SEED
}
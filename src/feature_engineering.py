# src/feature_engineering.py
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
import joblib
from typing import Tuple
from src import config, utils

logger = utils.get_logger(__name__)

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Builds the Scikit-learn pipeline for all feature transformations.
    This is the core of our feature engineering.
    """
    numeric_transformer = SimpleImputer(strategy='median')
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    high_cardinality_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('target_encoder', TargetEncoder(handle_missing='value', handle_unknown='value'))
    ])
    
    binary_transformer = SimpleImputer(strategy='constant', fill_value=0)
    
    # Remainder='passthrough' will keep any other features as they are
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, config.NUMERICAL_FEATURES),
            ('cat', categorical_transformer, config.CATEGORICAL_FEATURES),
            ('high_card', high_cardinality_transformer, config.HIGH_CARDINALITY_FEATURES),
            ('binary', binary_transformer, config.BINARY_FEATURES)
        ],
        remainder='drop',  # Drop any columns not explicitly listed
        verbose_feature_names_out=False
    )
    
    preprocessor.set_output(transform="pandas")  # Output a pandas DataFrame
    return preprocessor

def run_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies the full preprocessing pipeline to the data.
    Fits the pipeline and saves it for future use.
    """
    logger.info("Starting feature engineering...")
    
    # Initial data cleaning before pipeline
    df.drop(columns=config.FEATURES_TO_DROP, inplace=True, errors='ignore')
    df.dropna(subset=[config.TARGET_COLUMN], inplace=True)
    
    # Ensure all model features are present, fill missing ones with np.nan
    for col in config.MODEL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
            
    X = df[config.MODEL_FEATURES]
    y = np.log1p(df[config.TARGET_COLUMN])

    pipeline = build_preprocessing_pipeline()
    
    logger.info("Fitting feature engineering pipeline...")
    X_processed = pipeline.fit_transform(X, y)
    
    # Save the fitted pipeline
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    pipeline_path = os.path.join(config.MODELS_DIR, 'preprocessing_pipeline.pkl')
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Feature engineering pipeline saved to {pipeline_path}")
    
    return X_processed, y
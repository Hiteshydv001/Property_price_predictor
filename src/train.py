# src/train.py
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
from src import config, utils

logger = utils.get_logger(__name__)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating new geospatial features...")
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
    
    # --- THIS IS THE FIX ---
    # We no longer need to modify the config list here.
    # It's already defined correctly in config.py
    # -----------------------
    
    logger.info("... 'distance_to_center' feature created successfully.")
    return df

def run_hyperparameter_tuning_and_training():
    logger.info("--- Starting Advanced Model Training & Selection Pipeline ---")
    try:
        df = pd.read_csv(config.PROCESSED_MASTER_PATH, low_memory=False)
        logger.info(f"Successfully loaded data with {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"FATAL: Processed data not found. Run data processing first.")
        return

    df.dropna(subset=[config.TARGET_COLUMN, 'Latitude', 'Longitude', 'City'], inplace=True)
    df = add_geospatial_features(df)
    df.dropna(subset=['distance_to_center'], inplace=True)
    
    df_sample = df.sample(n=min(len(df), 50000), random_state=42)
    X = df_sample[config.MODEL_FEATURES]
    y = np.log1p(df_sample[config.TARGET_COLUMN])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Using sample of {len(df_sample)} rows. Split: {len(X_train)} train, {len(X_test)} test.")

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, config.NUMERICAL_FEATURES), ('cat', categorical_transformer, config.CATEGORICAL_FEATURES), ('target_loc', TargetEncoder(), config.HIGH_CARDINALITY_FEATURES)], remainder='passthrough')
    
    param_dist = {
        'preprocessor__num__imputer': [SimpleImputer(strategy='median')],
        'preprocessor__num__scaler': [StandardScaler(), 'passthrough'],
        'model': [LGBMRegressor(random_state=42, n_jobs=-1)],
        'model__n_estimators': [1000, 1500],
        'model__learning_rate': [0.01, 0.05],
        'model__num_leaves': [25, 31, 40],
        'model__max_depth': [7, 10],
        'model__reg_alpha': [0.1, 0.5, 1],
        'model__reg_lambda': [1, 2, 5],
        'model__colsample_bytree': [0.7, 0.8],
        'model__subsample': [0.7, 0.9]
    }
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LGBMRegressor())])
    
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=2, verbose=2, n_jobs=-1, scoring='r2', random_state=42)
    
    logger.info("Executing RandomizedSearchCV with new geospatial features...")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    end_time = time.time()
    
    logger.info(f"Search finished in {(end_time - start_time) / 60:.2f} minutes.")
    logger.info(f"Best cross-validated R² score found: {random_search.best_score_:.4f}")
    logger.info("Best parameters found:")
    for param, value in random_search.best_params_.items():
        logger.info(f"  {param}: {value}")
        
    best_pipeline = random_search.best_estimator_
    logger.info("\nTraining the best found pipeline on the full training data...")
    best_pipeline.fit(X_train, y_train)
    
    train_preds_log = best_pipeline.predict(X_train)
    test_preds_log = best_pipeline.predict(X_test)
    train_r2 = r2_score(y_train, train_preds_log)
    test_r2 = r2_score(y_test, test_preds_log)
    
    y_test_original = np.expm1(y_test)
    test_preds_original = np.expm1(test_preds_log)
    mae = mean_absolute_error(y_test_original, test_preds_original)

    logger.info("--- Final Model Performance ---")
    logger.info(f"Training R² Score: {train_r2:.4f}")
    logger.info(f"Test/Validation R² Score: {test_r2:.4f}")
    logger.info(f"Mean Absolute Error (MAE): ₹{mae:,.0f}")
    
    if train_r2 > test_r2 + 0.05:
        logger.warning("Overfitting still detected. Consider more feature engineering or stronger regularization.")
    else:
        logger.info("Model shows good generalization. Overfitting is controlled.")

    logger.info(f"\nSaving the best pipeline to {config.PIPELINE_PATH}")
    joblib.dump(best_pipeline, config.PIPELINE_PATH)
    logger.info("Pipeline saved successfully.")
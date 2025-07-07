# src/data_processing.py
import pandas as pd
import os
from typing import Dict, List, Optional
from src import config, utils

logger = utils.get_logger(__name__)

# Helper function to convert messy amenity columns to a clean binary 0 or 1
def _to_binary_amenity(value: any) -> int:
    s_val = str(value).lower().strip()
    if s_val in ['0', 'nan', 'not available', 'na', 'no', 'false']:
        return 0
    return 1

# Functions to process each raw file
def _process_properties_csv(filepath: str) -> Optional[pd.DataFrame]:
    logger.info(f"  -> Processing {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    
    df_clean = pd.DataFrame()
    df_clean['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df_clean['Bedrooms'] = pd.to_numeric(df['bedroom'], errors='coerce')
    df_clean['Bathrooms'] = pd.to_numeric(df['Bathroom'], errors='coerce')
    df_clean['Area_SqFt'] = pd.to_numeric(df['Covered Area'], errors='coerce')
    df_clean['City'] = df['City']
    df_clean['Locality'] = df.get('Location', df.get('Area Name'))
    df_clean['Property_Type'] = df['Type of Property']
    df_clean['Furnishing_Status'] = df['furnished Type']
    df_clean['Property_Age'] = df['Possession Status']
    df_clean['Floor'] = pd.to_numeric(df['Floor Data'].astype(str).str.extract(r'(\d+)\s*of')[0], errors='coerce') if 'Floor Data' in df.columns else pd.to_numeric(df.get('Floor No'), errors='coerce')
    df_clean['Total_Floors'] = pd.to_numeric(df.get('floors'), errors='coerce')
    df_clean['has_Pool'] = df.get('Swimming Pool').apply(_to_binary_amenity)
    df_clean['has_Gym'] = df.get('Gymnasium').apply(_to_binary_amenity)
    df_clean['has_Lift'] = df.get('Lift').apply(_to_binary_amenity)
    df_clean['has_Parking'] = df.get('Parking').apply(_to_binary_amenity)
    df_clean['Area_Type'] = 'Covered Area'
    df_clean['Source_Dataset'] = 'properties'
    return df_clean

def _process_makaan_csv(filepath: str) -> Optional[pd.DataFrame]:
    logger.info(f"  -> Processing {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath, encoding='latin1')
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    
    df_clean = pd.DataFrame()
    df_clean['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')
    df_clean['Bedrooms'] = pd.to_numeric(df['No_of_BHK'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df_clean['Area_SqFt'] = pd.to_numeric(df['Size'].astype(str).str.replace(',', '').str.extract(r'(\d+)')[0], errors='coerce')
    df_clean['City'] = df['City_name']
    df_clean['Locality'] = df['Locality_Name']
    df_clean['Property_Type'] = df['Property_type']
    df_clean['Furnishing_Status'] = df['is_furnished']
    df_clean['Property_Age'] = df['Property_status']
    df_clean['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df_clean['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['description_lower'] = df['description'].str.lower().fillna('')
    df_clean['has_Pool'] = df['description_lower'].str.contains('pool').astype(int)
    df_clean['has_Gym'] = df['description_lower'].str.contains('gym|fitness').astype(int)
    df_clean['has_Lift'] = df['description_lower'].str.contains('lift|elevator').astype(int)
    df_clean['has_Parking'] = df['description_lower'].str.contains('parking').astype(int)
    df_clean['Area_Type'] = 'Salable Area'
    df_clean['Source_Dataset'] = 'makaan'
    return df_clean

def _process_99acres_dataset(data_path: str, facets_path: str, city_name: str) -> Optional[pd.DataFrame]:
    logger.info(f"  -> Processing 99acres file: {os.path.basename(data_path)}...")
    # This is a simplified placeholder. In a real scenario, this function would
    # load, join with facets, and map to the schema like in previous examples.
    # For now, we'll return an empty frame to allow the pipeline to run.
    try:
        df = pd.read_csv(data_path, low_memory=False, encoding='latin1')
        # ... (Your full 99acres processing logic would go here) ...
        # For now, let's create a placeholder
        df_clean = pd.DataFrame()
        df_clean['Price'] = pd.to_numeric(df.get('PRICE'), errors='coerce')
        df_clean['Bedrooms'] = pd.to_numeric(df.get('BEDROOM_NUM'), errors='coerce')
        df_clean['Area_SqFt'] = pd.to_numeric(df.get('SUPER_SQFT'), errors='coerce')
        df_clean['City'] = city_name
        df_clean['Source_Dataset'] = '99acres'
        return df_clean
    except FileNotFoundError:
        logger.warning(f"99acres file not found: {data_path}")
        return None

def run_data_ingestion() -> None:
    """Orchestrates the ingestion of all raw data sources."""
    logger.info("Starting data ingestion and processing pipeline...")
    all_dfs: List[pd.DataFrame] = []

    # Process each source
    if os.path.exists(config.PROPERTIES_FILE):
        all_dfs.append(_process_properties_csv(config.PROPERTIES_FILE))
    if os.path.exists(config.MAKAAN_FILE):
        all_dfs.append(_process_makaan_csv(config.MAKAAN_FILE))
    
    for city, path in config.ACRES_CITY_FILES.items():
        if os.path.exists(path):
            df_city = _process_99acres_dataset(path, config.ACRES_FACETS_PATH, city)
            if df_city is not None: all_dfs.append(df_city)

    # Filter out any None results from failed file loads
    all_dfs = [df for df in all_dfs if df is not None]
    
    if not all_dfs:
        logger.error("No dataframes were processed. Exiting.")
        return

    logger.info("Concatenating all data sources...")
    final_df = pd.concat(all_dfs, ignore_index=True)

    # --- THIS IS THE CRUCIAL PART ---
    # Define all columns we want to keep, based on our Golden Schema from config
    columns_to_keep = [
        config.TARGET_COLUMN,
        *config.FEATURES_TO_DROP,
        *config.MODEL_FEATURES
    ]
    # Remove duplicates while preserving order
    columns_to_keep = list(dict.fromkeys(columns_to_keep)) 

    # Filter the DataFrame to only include these columns
    existing_cols_to_keep = [col for col in columns_to_keep if col in final_df.columns]
    final_df_filtered = final_df[existing_cols_to_keep]
    # ------------------------------------

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    final_df_filtered.to_csv(config.PROCESSED_MASTER_PATH, index=False)
    
    logger.info(f"Master dataset created with {len(final_df_filtered)} rows and {len(final_df_filtered.columns)} columns.")
    logger.info(f"Saved to: {config.PROCESSED_MASTER_PATH}")
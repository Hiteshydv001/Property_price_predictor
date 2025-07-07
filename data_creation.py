import pandas as pd
import os
import re
import ast # For safely evaluating string-formatted dictionaries

# --- Configuration: Define our target schema ---
GOLDEN_SCHEMA_COLUMNS = [
    'Price', 'Bedrooms', 'Bathrooms', 'Area_SqFt', 'Area_Type', 'City', 
    'Locality', 'Property_Type', 'Furnishing_Status', 'Property_Age', 'Floor', 
    'Total_Floors', 'Longitude', 'Latitude', 'has_Pool', 'has_Gym', 
    'has_Lift', 'has_Parking', 'Source_Dataset'
]

# --- CORRECTED Processing Function for properties.csv ---
def process_properties_csv(filepath):
    """Loads and transforms data from properties.csv to the golden schema."""
    print(f"Processing {filepath}...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        print(f"SKIPPED: {filepath} not found.")
        return None
    def to_binary_amenity(value):
        s_val = str(value).lower().strip()
        if s_val in ['0', 'nan', 'not available', 'na', 'no', 'false']:
            return 0
        return 1
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
    df_clean['has_Pool'] = df.get('Swimming Pool').apply(to_binary_amenity)
    df_clean['has_Gym'] = df.get('Gymnasium').apply(to_binary_amenity)
    df_clean['has_Lift'] = df.get('Lift').apply(to_binary_amenity)
    df_clean['has_Parking'] = df.get('Parking').apply(to_binary_amenity)
    df_clean['Area_Type'] = 'Covered Area'
    df_clean['Source_Dataset'] = 'properties'
    print(f"-> Successfully processed {len(df_clean)} rows from properties.csv")
    return df_clean

# --- CORRECTED Processing Function for Makaan_Properties_Buy.csv ---
def process_makaan_csv(filepath):
    """Loads and transforms data from Makaan_Properties_Buy.csv."""
    print(f"Processing {filepath}...")
    try:
        # --- THIS IS THE FIX ---
        # Specify the encoding to handle special characters
        df = pd.read_csv(filepath, encoding='latin1')
        # ----------------------
    except FileNotFoundError:
        print(f"SKIPPED: {filepath} not found.")
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
    print(f"-> Successfully processed {len(df_clean)} rows from Makaan_Properties_Buy.csv")
    return df_clean

# --- Processing Function for 99acres dataset (No changes needed here) ---
def process_99acres_dataset(data_path, facets_path, city_name):
    print(f"Processing 99acres file: {data_path} for City: {city_name}")
    try:
        df = pd.read_csv(data_path, low_memory=False, encoding='latin1')
        furnish_df = pd.read_csv(os.path.join(facets_path, 'FURNISH.csv'), encoding='latin1')
        prop_type_df = pd.read_csv(os.path.join(facets_path, 'PROPERTY_TYPE.csv'), encoding='latin1')
        amenities_df = pd.read_csv(os.path.join(facets_path, 'AMENITIES.csv'), encoding='latin1')
    except FileNotFoundError as e:
        print(f"SKIPPED: Could not find a required file for 99acres: {e}")
        return None
    df['FURNISH'] = df['FURNISH'].astype(str)
    furnish_df['id'] = furnish_df['id'].astype(str).str.lstrip('0')
    df = pd.merge(df, furnish_df, left_on='FURNISH', right_on='id', how='left', suffixes=('', '_furnish'))
    df['PROPERTY_TYPE'] = df['PROPERTY_TYPE'].astype(str)
    prop_type_df['id'] = prop_type_df['id'].astype(str).str.lstrip('0')
    df = pd.merge(df, prop_type_df, left_on='PROPERTY_TYPE', right_on='id', how='left', suffixes=('', '_proptype'))
    def extract_lat_lon(s):
        try:
            d = ast.literal_eval(s)
            return d.get('LATITUDE'), d.get('LONGITUDE')
        except (ValueError, SyntaxError, AttributeError):
            return None, None
    df[['LATITUDE_p', 'LONGITUDE_p']] = df['MAP_DETAILS'].apply(lambda x: pd.Series(extract_lat_lon(x)))
    amenities_map = amenities_df.set_index('id')['label']
    pool_id_series = amenities_df[amenities_df['label'].str.contains('Swimming Pool', case=False, na=False)]['id']
    pool_id = pool_id_series.iloc[0] if not pool_id_series.empty else -1
    gym_id_series = amenities_df[amenities_df['label'].str.contains('Fitness Centre / GYM', case=False, na=False)]['id']
    gym_id = gym_id_series.iloc[0] if not gym_id_series.empty else -1
    lift_id_series = amenities_df[amenities_df['label'].str.contains('Lift', case=False, na=False)]['id']
    lift_id = lift_id_series.iloc[0] if not lift_id_series.empty else -1
    parking_id_series = amenities_df[amenities_df['label'].str.contains('Security / Fire Alarm', case=False, na=False)]['id']
    parking_id = parking_id_series.iloc[0] if not parking_id_series.empty else -1
    df['FEATURES_str'] = df.get('FEATURES', '').astype(str)
    df['has_Pool'] = df['FEATURES_str'].str.contains(str(pool_id)).astype(int)
    df['has_Gym'] = df['FEATURES_str'].str.contains(str(gym_id)).astype(int)
    df['has_Lift'] = df['FEATURES_str'].str.contains(str(lift_id)).astype(int)
    df['has_Parking'] = df['FEATURES_str'].str.contains(str(parking_id)).astype(int)
    df_clean = pd.DataFrame()
    df_clean['Price'] = pd.to_numeric(df.get('PRICE'), errors='coerce')
    df_clean['Bedrooms'] = pd.to_numeric(df.get('BEDROOM_NUM'), errors='coerce')
    df_clean['Bathrooms'] = pd.to_numeric(df.get('BATHROOM_NUM'), errors='coerce')
    df_clean['Area_SqFt'] = pd.to_numeric(df.get('SUPER_SQFT'), errors='coerce')
    df_clean['City'] = city_name
    df_clean['Locality'] = df.get('PROP_HEADING', '').astype(str).str.split(' in ').str[1].str.split(',').str[0]
    df_clean['Property_Type'] = df['label_proptype']
    df_clean['Furnishing_Status'] = df['label']
    df_clean['Property_Age'] = df['AGE']
    df_clean['Floor'] = pd.to_numeric(df.get('FLOOR_NUM'), errors='coerce')
    df_clean['Total_Floors'] = pd.to_numeric(df.get('TOTAL_FLOOR'), errors='coerce')
    df_clean['Latitude'] = pd.to_numeric(df['LATITUDE_p'], errors='coerce')
    df_clean['Longitude'] = pd.to_numeric(df['LONGITUDE_p'], errors='coerce')
    df_clean['has_Pool'] = df['has_Pool']
    df_clean['has_Gym'] = df['has_Gym']
    df_clean['has_Lift'] = df['has_Lift']
    df_clean['has_Parking'] = df['has_Parking']
    df_clean['Area_Type'] = 'Super Built-up'
    df_clean['Source_Dataset'] = '99acres'
    print(f"-> Successfully processed {len(df_clean)} rows.")
    return df_clean

# --- Main Execution Block ---
def main():
    properties_file = 'properties.csv'
    makaan_file = 'Makaan_Properties_Buy.csv'
    acres_folder = '99acre'
    acres_facets_path = os.path.join(acres_folder, 'facets', 'facets')
    acres_city_files = {
        'Gurugram': os.path.join(acres_folder, 'gurgaon_10k.csv'),
        'Hyderabad': os.path.join(acres_folder, 'hyderabad.csv'),
        'Kolkata': os.path.join(acres_folder, 'kolkata.csv'),
        'Mumbai': os.path.join(acres_folder, 'mumbai.csv')
    }
    all_dfs = []
    df1 = process_properties_csv(properties_file)
    if df1 is not None: all_dfs.append(df1)
    df2 = process_makaan_csv(makaan_file)
    if df2 is not None: all_dfs.append(df2)
    for city, path in acres_city_files.items():
        if os.path.exists(path):
            df_city = process_99acres_dataset(path, acres_facets_path, city)
            if df_city is not None: all_dfs.append(df_city)
        else:
            print(f"SKIPPED: {path} not found.")

    if not all_dfs:
        print("\nNo datasets were processed. Please check file paths and errors.")
        return
    print("\nCombining all processed datasets...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.reindex(columns=GOLDEN_SCHEMA_COLUMNS)
    output_filename = 'master_india_properties.csv'
    final_df.to_csv(output_filename, index=False)
    print("\n" + "="*40)
    print("      SCRIPT COMPLETE      ")
    print("="*40)
    print(f"Successfully created '{output_filename}'")
    print(f"Total Rows: {len(final_df)}")
    print(f"Total Columns: {len(final_df.columns)}")
    print("\nFinal DataFrame Info:")
    final_df.info()
    print("\nSample of the final data:")
    print(final_df.head())

if __name__ == '__main__':
    main()
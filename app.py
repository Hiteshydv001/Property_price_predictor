import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import os
import logging

# --- Setup ---
st.set_page_config(page_title="Commercial Property Price Estimator", page_icon="🏢", layout="centered")
logging.basicConfig(level=logging.INFO)

# --- Load Models and Data (Cached) ---
@st.cache_resource
def load_models():
    """Loads all K-Fold models from the 'models' directory."""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    models = []
    for i in range(5): # Assuming 5 folds
        model_path = os.path.join(models_dir, f'property_price_prod_model_fold_{i+1}.json')
        if not os.path.exists(model_path):
            return None # Models not found
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        models.append(model)
    return models

@st.cache_data
def load_processed_data_for_ui():
    """Loads data to populate UI elements and for feature engineering."""
    data_path = os.path.join('data', 'processed', 'master_india_properties.csv')
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path, low_memory=False)
    return df

models = load_models()
df_for_ui = load_processed_data_for_ui()

# --- App Header ---
st.title("Enterprise-Grade Property Price Estimator 🏢")
st.markdown("Enter property details to get an AI-powered price estimation using an ensemble of 5 advanced models.")

if models is None or df_for_ui is None:
    st.error("Model files or processed data not found. Please ensure the training pipeline has been run successfully.")
else:
    def create_property_form():
        # --- UI Inputs ---
        with st.form("prediction_form"):
            st.header("Property Details")
            
            # Get unique, sorted lists for dropdowns
            if 'city' not in df_for_ui.columns:
                st.error("City information is missing in the processed data. Please run the data processing pipeline again.")
                return None
                
            top_cities = sorted(df_for_ui['city'].value_counts().nlargest(20).index.tolist())
            if not top_cities:  # If no cities found
                st.error("No city data available. Please check the data processing pipeline.")
                return None
                
            prop_types = sorted(df_for_ui['property_type'].unique().tolist()) if 'property_type' in df_for_ui.columns else ['Other']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                city = st.selectbox("📍 City", options=top_cities)
            with col2:
                prop_type = st.selectbox("🏗️ Property Type", options=prop_types)
            with col3:
                area_sqft = st.number_input("📐 Area (sq. ft.)", min_value=100, max_value=50000, value=1200, step=50)

            # Dynamic locality dropdown based on selected city
            localities_in_city = sorted(df_for_ui[df_for_ui['city'] == city]['locality'].unique().tolist())
            locality = st.selectbox("🏘️ Locality", options=localities_in_city)
            
            bedrooms = st.slider("🛏️ Bedrooms (if applicable)", 0, 10, 0)

            st.write("---")
            submitted = st.form_submit_button("Predict Price")
            
            if submitted:
                return {
                    'city': city,
                    'property_type': prop_type,
                    'area_sqft': area_sqft,
                    'locality': locality,
                    'bedrooms': bedrooms
                }
            return None

    # Get form data
    form_data = create_property_form()

    # --- Prediction Logic ---
    if form_data:
        with st.spinner("Running property valuation through the model ensemble..."):
            # 1. Create a base DataFrame for input
            input_df = pd.DataFrame([{
                'area_sqft': float(form_data['area_sqft']),
                'bedrooms': float(form_data['bedrooms']),
                'log_area': np.log1p(form_data['area_sqft'])
            }])

            # 2. Add one-hot encoded features
            city = form_data['city']
            prop_type = form_data['property_type']
            top_cities = sorted(df_for_ui['city'].value_counts().nlargest(20).index.tolist())
            prop_types = sorted(df_for_ui['property_type'].unique().tolist())
            
            for c in top_cities[1:]: # drop_first=True equivalent
                input_df[f'city_{c}'] = 1 if city == c else 0
            for pt in prop_types[1:]:
                input_df[f'property_type_{pt}'] = 1 if prop_type == pt else 0

            # 3. Add target-encoded features (using stats from the full dataset for simplicity in UI)
            locality_stats = df_for_ui.groupby('locality').agg(
                locality_log_price_mean=('log_price', 'mean'),
                locality_pps_mean=('price_per_sqft', 'mean')
            ).reset_index()
            
            selected_locality_stats = locality_stats[locality_stats['locality'] == locality]
            
            if not selected_locality_stats.empty:
                input_df['locality_log_price_mean'] = selected_locality_stats['locality_log_price_mean'].values[0]
                input_df['locality_pps_mean'] = selected_locality_stats['locality_pps_mean'].values[0]
            else: # Fallback for new localities
                input_df['locality_log_price_mean'] = df_for_ui['log_price'].mean()
                input_df['locality_pps_mean'] = df_for_ui['price_per_sqft'].mean()

            # 4. Make Predictions with the Ensemble
            all_predictions_log = []
            model_features = models[0].get_booster().feature_names
            
            # Ensure input_df has all the model's required columns in the correct order
            for col in model_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model_features]

            for model in models:
                pred_log = model.predict(input_df)
                all_predictions_log.append(pred_log[0])
            
            # Average the log predictions and then inverse transform
            avg_log_prediction = np.mean(all_predictions_log)
            final_prediction = np.expm1(avg_log_prediction)

        # --- Display Result ---
        st.success("Valuation Complete!")
        
        if final_prediction > 1_00_00_000:
            price_display = f"₹ {final_prediction / 1_00_00_000:.2f} Crore"
        else:
            price_display = f"₹ {final_prediction / 1_00_000:.2f} Lakhs"
            
        st.metric(label="AI-Powered Price Estimate", value=price_display)
        st.info("This estimate is the average from 5 models trained on different parts of the data, providing a more robust valuation.")
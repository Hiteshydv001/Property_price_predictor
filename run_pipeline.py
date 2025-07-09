# run_pipeline.py
import os
from src import data_processing, train, utils, config

logger = utils.get_logger(__name__)

def main():
    """Orchestrates the entire ML pipeline."""
    
    logger.info("--- Starting Master ML Pipeline ---")
    
    # Step 1: Ingest and process raw data
    logger.info("="*20 + " [1/2] Data Processing " + "="*20)
    data_processing.run_data_ingestion()
    
    # Step 2: Run hyperparameter tuning and train the best model
    logger.info("="*20 + " [2/2] Model Selection & Training " + "="*23)
    train.run_hyperparameter_tuning_and_training()
    
    logger.info("--- Master ML Pipeline Finished Successfully! ---")
    
    # --- THIS IS THE FIX ---
    # The variable was named MODEL_DIR in the config file.
    logger.info(f"Best pipeline artifact saved in '{config.MODEL_DIR}'")
    # -----------------------

if __name__ == "__main__":
    if not os.path.basename(os.getcwd()) == 'property_price_predictor':
         print(f"ERROR: This script must be run from the root of the 'property_price_predictor' directory, but you are in '{os.getcwd()}'")
         exit()
         
    main()
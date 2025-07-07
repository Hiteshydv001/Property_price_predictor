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
    
    # Step 2: Train the model (this step also runs feature engineering)
    logger.info("="*20 + " [2/2] Model Training " + "="*23)
    train.run_training()
    
    logger.info("--- Master ML Pipeline Finished Successfully! ---")
    logger.info(f"All artifacts saved in '{config.MODELS_DIR}'")

if __name__ == "__main__":
    # This check ensures the script is run from the correct directory
    if not os.path.basename(os.getcwd()) == 'property_price_predictor':
         print(f"ERROR: This script must be run from the root of the 'property_price_predictor' directory, but you are in '{os.getcwd()}'")
         exit()
         
    main()

---

# Real Estate Price Prediction Engine

This project contains a complete, industry-grade machine learning pipeline for predicting property prices across various cities in India. The system ingests raw data from multiple sources, performs advanced feature engineering and hyperparameter tuning to find the best model, and serves predictions via a simple web interface.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Final Model Performance](#final-model-performance-metrics)
3.  [Tech Stack](#tech-stack)
4.  [Project Structure](#project-structure)
5.  [Setup and Installation](#setup-and-installation)
6.  [How to Run the Pipeline](#how-to-run-the-pipeline)
7.  [How to Use the Prediction App](#how-to-use-the-prediction-app)
8.  [Future Improvements](#future-improvements)

## Project Overview

The primary goal of this project is to build a highly accurate and reliable model to estimate the market value of real estate properties. The pipeline is designed to be robust, automated, and easily extensible.

**Key Features:**
- **Automated Data Ingestion:** Processes and combines data from multiple raw sources (`properties.csv`, `Makaan`, `99acres`).
- **Advanced Feature Engineering:** Automatically creates powerful features like `distance_to_city_center` from geospatial data.
- **Systematic Model Selection:** Uses `RandomizedSearchCV` to test various models (LightGBM, XGBoost), preprocessing techniques (scaling, imputation), and hyperparameters to find the optimal combination.
- **Overfitting Control:** Employs strong regularization techniques and evaluates the model's generalization capability by comparing training and validation scores.
- **Saved Artifacts:** The entire winning pipeline (including preprocessing steps and the trained model) is saved as a single artifact for consistent predictions.
- **Interactive Web UI:** A simple Flask application allows for real-time model testing and demonstration.

## Final Model Performance Metrics

The pipeline has been successfully executed, and the following results were achieved for the best-performing model.

### 1. The Winning Pipeline
- **Model Algorithm:** **LightGBM (LGBMRegressor)**
- **Numerical Imputation:** `KNNImputer`
- **Numerical Scaling:** `StandardScaler`
- **Key Feature Added:** `distance_to_center` (geospatial feature)

### 2. Prediction Accuracy
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Test/Validation R-squared (R²)** | **0.9034** | The model explains **90.3%** of the price variation on new, unseen properties, meeting our primary target. |
| **Mean Absolute Error (MAE)** | **₹28,87,156** | On average, the model's price prediction is off by approximately **₹29 Lakhs**. |

### 3. Model Reliability
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Training R²** | 0.9461 | How well the model performed on data it was trained on. |
| **Test R²** | 0.9034 | How well the model performed on new data. |
| **Overfitting Gap** | 4.27% | The small gap indicates that **the model generalizes well** and is reliable for production use. |

## Tech Stack
- **Language:** Python 3.10+
- **Core Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Category Encoders
- **Web Framework:** Flask
- **Environment Management:** `venv`

## Project Structure
```
/property_price_predictor/
|
|-- app/                  # Contains the Flask web application for the UI
|   |-- templates/
|   |   |-- index.html
|   |-- main.py
|
|-- data/                 # All project data
|   |-- raw/              # Raw, immutable data sources
|   |-- processed/        # Cleaned and processed data files
|
|-- models/               # Saved model artifacts (e.g., .pkl, .json)
|
|-- notebooks/            # Jupyter notebooks for exploration and prototyping
|
|-- src/                  # Source code for the ML pipeline
|   |-- config.py         # Central configuration file
|   |-- data_processing.py # Data ingestion and cleaning scripts
|   |-- feature_engineering.py # Feature creation and transformation
|   |-- predict.py        # Logic for making predictions with a saved model
|   |-- train.py          # Model training and tuning logic
|   |-- utils.py          # Utility functions like logging
|
|-- .gitignore            # Files to be ignored by Git
|-- requirements.txt      # Project dependencies
|-- run_pipeline.py       # Master script to run the full training pipeline
```

## How to Run the Pipeline

The master script `run_pipeline.py` executes the entire workflow from data ingestion to model training and saving.

**To run the full pipeline, execute the following command from the root `property_price_predictor` directory:**

```bash
python run_pipeline.py
```

**What this script does:**
1.  **Stage 1: Data Processing:** It reads all raw CSV files from `data/raw/`, cleans them, combines them into a single master file, and saves it to `data/processed/`.
2.  **Stage 2: Model Selection & Training:** It loads the master file, engineers advanced features, runs `RandomizedSearchCV` to find the best model and preprocessing steps, trains the final pipeline, evaluates it, and saves the resulting artifact (`preprocessing_pipeline.pkl`) to the `models/` directory.

> **Note:** The hyperparameter search can be computationally intensive. For a quick test run, you can reduce the `n_iter` and `cv` values in `src/train.py`.

## How to Use the Prediction App

After the pipeline has been run successfully at least once, the trained model artifact will be available. You can then launch the web UI to test it.

1.  **Start the Flask Server:** Run the following command from the root `property_price_predictor` directory:
    ```bash
    python app/main.py
    ```

2.  **Open Your Browser:** Navigate to the following address:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

3.  **Get a Prediction:**
    - The web page will display a form.
    - Select a `City` from the dropdown. The `Locality` dropdown will automatically update with the relevant options for that city.
    - Fill in the other details and click "Estimate Price".
    - The model's prediction will be displayed on the page.

## Future Improvements

- **Expand Feature Engineering:** Add more aggregate features (e.g., `median_price_in_locality`) and time-based features to capture market trends.
- **Stacking Ensemble:** Implement a `StackingRegressor` with LightGBM, XGBoost, and CatBoost as base models to potentially push the R² score even higher.
- **CI/CD Pipeline:** Set up a Continuous Integration/Continuous Deployment pipeline (e.g., using GitHub Actions) to automatically retrain and deploy the model when new data is available.
- **Enhanced UI:** Improve the web interface with more interactive elements, such as a map to select location and charts to display prediction confidence intervals.

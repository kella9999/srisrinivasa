# README.md

# Crypto Price Movement Prediction API

This project is a machine learning-powered API designed to predict the short-term price movement of Bitcoin (BTC). It uses an XGBoost model trained on historical price data and a variety of technical indicators.

The API exposes a single endpoint, `/predict`, that accepts the latest closing price and volume to forecast whether the price is likely to move up or down in the next period.

## Core Components

-   **`app.py`**: The main Flask application that serves the prediction API.
-   **`core/predictor.py`**: A class that loads the trained model and handles the prediction logic.
-   **`models/train.py`**: The script used for downloading data, training the XGBoost model, and saving the final model and data scaler.
-   **`utils/feature_engineer.py`**: A centralized module for generating the 30 technical features required by the model.

## How to Run

### 1. Install Dependencies
Ensure you have Python 3.10 installed. Then, install the required packages:
```bash
pip install -r requirements.txt

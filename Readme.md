# Global Tech Stock Analysis & Prediction

This project analyzes historical stock data and financial metrics of global technology leaders (e.g., ASML, TSMC, Sony) using Python. It includes a dual-purpose pipeline for **price prediction (regression)** and **performance categorization (classification)**.

## 🚀 Features
- **Data Cleaning:** Handles missing values and merges global market datasets.
- **Price Forecasting:** Uses a Random Forest Regressor to predict next-day closing prices based on moving averages and historical trends.
- **Company Profiling:** Uses a Random Forest Classifier to categorize stocks based on 5-year performance metrics (PE Ratio, ROE, Beta, etc.).
- **Automated Reporting:** Generates a CSV of predictions for further analysis.

## 🛠️ Requirements
Ensure you have Python 3.8+ installed. You will need the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

You can install them via pip:
```bash
pip install numpy pandas scikit-learn matplotlib


├── main.py               # Primary script for data loading, ML modeling, and evaluation
├── Global_Tech_Historical_US_Market_*.csv  # Historical price data (US Market)
├── Global_Tech_Leaders_Stock_Dataset_*.csv # Fundamental metrics & company profiles
└── asml_predictions.csv  # [Generated] CSV containing actual vs. predicted prices


## 📋 Usage
1. Ensure your datasets are in the root folder.
2. Execute the script:
   ```bash
   python main.py
    ```

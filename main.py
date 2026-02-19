import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Get absolute path for file operations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_analysis():
    # 1. Load the Datasets
    try:
        us_hist = pd.read_csv(os.path.join(BASE_DIR, 'Global_Tech_Historical_US_Market_20260206_045434.csv'))
        leaders = pd.read_csv(os.path.join(BASE_DIR, 'Global_Tech_Leaders_Stock_Dataset_20260206_045419.csv'))
        print("Datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Make sure all CSV files are in the same directory as main.py")
        return
    except PermissionError as e:
        print(f"Error: Permission denied - {e}")
        print("Please check file permissions and try running as administrator")
        return
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # --- PART 1: Regression (Predicting Stock Price) ---
    # We will predict the next day's 'Close' price for ASML using historical US data.
    print("\n--- Task 1: Regression (ASML Price Prediction) ---")
    
    # Filter for ASML and prepare time-series features
    asml_data = us_hist[us_hist['Company'].str.contains('ASML')].copy()
    asml_data['Date'] = pd.to_datetime(asml_data['Date'])
    asml_data = asml_data.sort_values('Date')

    # Feature Engineering: Use previous day's price and a 5-day moving average
    asml_data['Prev_Close'] = asml_data['Close'].shift(1)
    asml_data['MA_5'] = asml_data['Close'].rolling(window=5).mean()
    asml_data.dropna(inplace=True)

    # Define Features (X) and Target (y)
    X_reg = asml_data[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'MA_5']]
    y_reg = asml_data['Close']

    # Split data (Sequential split for time-series)
    split_idx = int(len(X_reg) * 0.8)
    X_train_r, X_test_r = X_reg[:split_idx], X_reg[split_idx:]
    y_train_r, y_test_r = y_reg[:split_idx], y_reg[split_idx:]

    # Train Random Forest Regressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train_r, y_train_r)
    
    # Predict and Evaluate
    y_pred_r = regressor.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    print(f"ASML Stock Price Prediction RMSE: ${rmse:.2f}")

    # Plot actual vs predicted for the test period (uses matplotlib)
    try:
        dates_test = asml_data['Date'].iloc[split_idx:]
        plt.figure(figsize=(10, 5))
        plt.plot(dates_test, y_test_r.values, label='Actual Close')
        plt.plot(dates_test, y_pred_r, label='Predicted Close')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('ASML Actual vs Predicted Close Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'asml_actual_vs_predicted.png'))
        plt.close()
        print("Saved plot to 'asml_actual_vs_predicted.png'.")
    except PermissionError as e:
        print(f"Warning: Permission denied saving plot - {e}")
    except Exception as e:
        # plotting is optional; don't fail the whole run if it errors
        print(f"Warning: Could not save plot - {e}")

    # --- PART 2: Classification (Predicting Performance Categories) ---
    # We will use financial metrics to predict the '5Y_Performance_Category_US'.
    print("\n--- Task 2: Classification (Performance Category) ---")
    
    # Selecting relevant financial features
    features = ['PE_Ratio', 'Price_to_Book', 'Profit_Margins', 'Operating_Margins', 'ROE', 'Beta']
    target = '5Y_Performance_Category_US'

    # Clean the dataset for classification
    df_clf = leaders[features + [target]].dropna()
    
    X_clf = df_clf[features]
    y_clf = df_clf[target]

    # Encode the categorical target labels
    le = LabelEncoder()
    y_clf_encoded = le.fit_transform(y_clf)

    # Split and Scale
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf_encoded, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_c_scaled = scaler.fit_transform(X_train_c)
    X_test_c_scaled = scaler.transform(X_test_c)

    # Train Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_c_scaled, y_train_c)
    
    # Evaluate
    y_pred_c = classifier.predict(X_test_c_scaled)
    acc = accuracy_score(y_test_c, y_pred_c)
    print(f"Performance Category Classification Accuracy: {acc * 100:.2f}%")

    # Export a summary of processed data
    try:
        asml_data['Predicted_Close'] = np.nan
        asml_data.iloc[split_idx:, asml_data.columns.get_loc('Predicted_Close')] = y_pred_r
        asml_data.to_csv(os.path.join(BASE_DIR, 'asml_predictions.csv'), index=False)
        print("\nProcessing complete. Predictions saved to 'asml_predictions.csv'.")
    except PermissionError as e:
        print(f"Error: Permission denied saving CSV - {e}")
        print("Please check file permissions and try running as administrator")
    except Exception as e:
        print(f"Error saving predictions: {e}")

if __name__ == "__main__":
    run_analysis()

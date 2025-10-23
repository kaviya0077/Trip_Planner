import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json
import os
import re # <-- IMPORT THE 're' LIBRARY

def train_and_save_model(dataset_path="data/trips_dataset.csv"):
    """Loads data, prepares it, trains a model, and saves it."""
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"Creating directory: {models_dir}")
        os.makedirs(models_dir)

    # 1. Load and Prepare Data
    print("Loading and preparing data...")
    df = pd.read_csv(dataset_path)
    
    target_cols = ['accommodation_cost', 'food_cost', 'fuel_cost', 'activities_cost']
    df.dropna(subset=target_cols, inplace=True)
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_encoded = pd.get_dummies(X, columns=['start_city', 'end_city', 'budget_level', 'vehicle_type'], drop_first=True)
    
    # ▼▼▼ NEW CODE BLOCK TO FIX THE ERROR ▼▼▼
    # Clean column names to be compatible with LightGBM
    print("Cleaning feature names for LightGBM compatibility...")
    X_encoded.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X_encoded.columns]
    # ▲▲▲ END OF NEW CODE BLOCK ▲▲▲

    # Save the CLEANED columns used for training
    training_columns = X_encoded.columns.tolist()
    with open('models/training_columns.json', 'w') as f:
        json.dump(training_columns, f)

    # (The rest of the function remains the same)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    print("Training the LightGBM model...")
    lgbm = lgb.LGBMRegressor(random_state=42)
    model = MultiOutputRegressor(lgbm)
    model.fit(X_train, y_train)

    print("Evaluating the model...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error on Test Set: ₹{mae:,.2f}")

    print("Saving the model...")
    joblib.dump(model, 'models/cost_model.joblib')
    print("Model saved as models/cost_model.joblib")

if __name__ == "__main__":
    train_and_save_model()
import pandas as pd
import joblib
import json
import re
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px

def evaluate_model(model_path='models/cost_model.joblib', 
                   columns_path='models/training_columns.json', 
                   dataset_path='data/trips_dataset.csv'):
    """
    Loads the trained model and evaluates its performance against the dataset.
    """
    
    # 1. Load Model and Data
    print("--- Loading Model and Data ---")
    try:
        model = joblib.load(model_path)
        with open(columns_path, 'r') as f:
            training_columns = json.load(f)
        df = pd.read_csv(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return

    # 2. Prepare Data (same logic as in main.py and train_model.py)
    target_cols = ['accommodation_cost', 'food_cost', 'fuel_cost', 'activities_cost']
    df.dropna(subset=target_cols, inplace=True)
    X = df.drop(columns=target_cols)
    y_true = df[target_cols] # The "correct answers" from the dataset

    X_encoded = pd.get_dummies(X, columns=['start_city', 'end_city', 'budget_level', 'vehicle_type'], drop_first=True)
    X_encoded.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X_encoded.columns]
    X_aligned = X_encoded.reindex(columns=training_columns, fill_value=0)

    # 3. Make Predictions on the ENTIRE dataset
    print("\n--- Making Predictions ---")
    y_pred = model.predict(X_aligned)
    y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_true.index)

    # 4. Calculate and Print Performance Metrics
    print("\n--- Model Performance Metrics ---")
    
    # R-squared: "How much of the variation in cost can our model explain?"
    # A score of 1.0 is perfect. A score of 0.7 is generally considered good.
    r2 = r2_score(y_true, y_pred)
    print(f"R-squared Score (Overall): {r2:.2f}")
    
    # Mean Absolute Error (MAE): "On average, how much is our prediction off by?"
    mae_overall = mean_absolute_error(y_true, y_pred)
    print(f"Mean Absolute Error (Overall): ₹{mae_overall:,.0f}")

    print("\n--- Per-Category MAE ---")
    for i, col in enumerate(target_cols):
        mae_col = mean_absolute_error(y_true[col], y_pred_df[col])
        print(f"  - {col.title()}: ₹{mae_col:,.0f}")
        
    # 5. Find the Worst Predictions
    print("\n--- Analyzing Worst Predictions ---")
    df['predicted_total_cost'] = y_pred_df.sum(axis=1)
    df['actual_total_cost'] = y_true.sum(axis=1)
    df['error'] = abs(df['predicted_total_cost'] - df['actual_total_cost'])
    
    worst_predictions = df.sort_values(by='error', ascending=False).head(5)
    
    print("Top 5 trips where the model had the largest error:")
    for index, row in worst_predictions.iterrows():
        print(f"  - Trip: {row['start_city']} to {row['end_city']} ({row['budget_level']})")
        print(f"    Predicted Cost: ₹{row['predicted_total_cost']:,.0f}, Actual (LLM) Cost: ₹{row['actual_total_cost']:,.0f}, Error: ₹{row['error']:,.0f}\n")

if __name__ == "__main__":
    evaluate_model()
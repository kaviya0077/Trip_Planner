import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json
import os
import re
import optuna

def prepare_data(dataset_path="data/trips_dataset.csv"):
    """Loads and prepares the dataset, returning features and targets."""
    df = pd.read_csv(dataset_path)
    target_cols = ['accommodation_cost', 'food_cost', 'fuel_cost', 'activities_cost']
    df.dropna(subset=target_cols, inplace=True)
    
    X = df.drop(columns=target_cols)
    y = df[target_cols]
    
    X_encoded = pd.get_dummies(X, columns=['start_city', 'end_city', 'budget_level', 'vehicle_type'], drop_first=True)
    X_encoded.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X_encoded.columns]
    
    # Save the cleaned columns
    with open('models/training_columns.json', 'w') as f:
        json.dump(X_encoded.columns.tolist(), f)
        
    return X_encoded, y

def objective(trial, X, y):
    """
    This is the function that Optuna tries to minimize.
    It trains a model with a given set of hyperparameters and returns its error.
    """
    # Split data into training and validation sets for this trial
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Define the hyperparameter search space for Optuna
    params = {
        'objective': 'regression_l1',  # L1 is Mean Absolute Error, our target metric
        'metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'random_state': 42,
        'n_jobs': -1, # Use all available CPU cores
        'verbose': -1 # Suppress verbose output
    }

    # 2. Train a model with the suggested parameters
    lgbm = lgb.LGBMRegressor(**params)
    model = MultiOutputRegressor(lgbm)
    model.fit(X_train, y_train)

    # 3. Evaluate the model on the validation set
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)

    # 4. Return the error score for Optuna to minimize
    return mae

# In tune_model.py

def main():
    """Main function to run the tuning process and save the best model."""
    
    # Ensure the 'models' directory exists
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print("Loading and preparing data...")
    X, y = prepare_data()
    
    # Create an Optuna study. 'minimize' means we want the lowest MAE.
    study = optuna.create_study(direction='minimize')
    
    # Start the optimization process. 'run' is changed to 'optimize'.
    print("\nStarting hyperparameter tuning...")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    print("\nTuning finished!")
    print(f"Best trial MAE: ₹{study.best_value:,.2f}")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
        
    # --- Train the final model with the BEST hyperparameters on ALL data ---
    print("\nTraining final model with the best parameters on the full dataset...")
    
    best_params = study.best_params
    best_params['objective'] = 'regression_l1'
    best_params['metric'] = 'mae'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    final_lgbm = lgb.LGBMRegressor(**best_params)
    final_model = MultiOutputRegressor(final_lgbm)
    final_model.fit(X, y) # Train on 100% of the data

    # Save the final, tuned model
    joblib.dump(final_model, 'models/cost_model.joblib')
    print("\nFinal, tuned model saved successfully to models/cost_model.joblib")

if __name__ == "__main__":
    main()
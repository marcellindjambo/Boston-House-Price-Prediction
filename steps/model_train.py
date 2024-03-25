import os
import joblib
import mlflow
import pandas as pd
from src.build_model import RandomForestModel, KNeighborsRegressorModel, DecisionTreeRegressorModel

# Configure MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("House Price Prediction")

def train_evaluate_and_save_all_models(X_train, X_test, y_train, y_test):
    """
    Trains multiple regression models, evaluates them, and saves all models.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training targets.
        y_test (pd.Series): Testing targets.
    """
    # Initialize models
    models = {
        'Random Forest': RandomForestModel(),
        'Decision Tree': DecisionTreeRegressorModel(),
        'KNN': KNeighborsRegressorModel()
    }

    # Train, evaluate, and log all models
    for model_name, model in models.items():
        trained_model = model.train(X_train, y_train)
        y_pred = model.predict(trained_model, X_test)
        score, mse, mae, max_err = model.evaluate(y_pred, y_test)

        # Log metrics and model
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("Model Name", model_name)
            mlflow.log_metrics({
                "R2 Score": score,
                "Mean Squared Error": mse,
                "Mean Absolute Error": mae,
                "Max Error": max_err
            })
            
            mlflow.sklearn.log_model(model, "model")

        print(f"{model_name}: R2 Score - {score} | MSE - {mse} | MAE - {mae} | Max Error - {max_err}")

        # Save the model
        save_model_path = "model_saved"
        os.makedirs(save_model_path, exist_ok=True)
        model_file_path = os.path.join(save_model_path, f"{model_name}_model.pkl")
        joblib.dump(trained_model, model_file_path)

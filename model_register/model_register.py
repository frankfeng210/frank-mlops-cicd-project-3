
import os
import argparse
import logging
import mlflow
import pandas as pd
from pathlib import Path

def main():
    # Argument parser setup for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the trained model")  # Path to the trained model artifact
    args = parser.parse_args()

    mlflow.start_run()                                                  # Starting the MLflow experiment run

    print(f"Best-trained-model-name: {args.model}")

    # Load the trained model from the provided path
    #sk_model = mlflow.sklearn.load_model(Path(args.model))
    sk_model = mlflow.sklearn.load_model(args.model)

    print("Registering the best trained prediction model for used cars price prediction")

    # Register the model in the MLflow Model Registry under the name "used_cars_price_prediction_model"
    mlflow.sklearn.log_model(
        sk_model=sk_model,
        registered_model_name="used_cars_price_prediction_model",       # Descriptive model name for registration
        artifact_path="random_forest_price_regressor"                   # Path to store model artifacts
    )

    # End the MLflow run
    mlflow.end_run()

if __name__ == "__main__":
    main()

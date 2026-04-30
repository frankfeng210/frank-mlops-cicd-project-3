
###################################################
# train regression model without parameter tuning #
###################################################

# Required imports for training
import mlflow
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

model_output_dir = "./outputs"
os.makedirs(model_output_dir, exist_ok=True)        # Create the "outputs" directory if it doesn't exist
model_path = os.path.abspath(model_output_dir)

def select_first_file(path):
    """Selects the first file in a folder, assuming there's only one file.
    Args:
        path (str): Path to the directory or file to choose.
    Returns:
        str: Full path of the selected file.
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',   type=str,               help='Path to train data.')
    parser.add_argument("--test_data",    type=str,               help='Path to test data.')
    parser.add_argument('--n_estimators', type=int, default=100,  help='The number of decision trees in the forest.')
    parser.add_argument('--max_depth',    type=int, default=None, help='The maximum depth of the decision tree.')
    parser.add_argument('--model_output', type=str,               help='Path of output model.')
    args = parser.parse_args()

    mlflow.start_run()                                  # Start the MLflow experiment run

    # 1. Load datasets
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df  = pd.read_csv(select_first_file(args.test_data))
  
    # 2. prepare train & test datasets

    # Dropping the label column and assigning it to y_train
    y_train = train_df["price"].values                  # vehicle 'price' is the target variable in this case study
    X_train = train_df.drop("price", axis=1).values     # dropping the 'price' column from train_df to get the features and converting to array for model training

    # Dropping the label column and assigning it to y_test
    y_test = test_df["price"].values                    # vehicle 'price' is the target variable for testing
    X_test = test_df.drop("price", axis=1).values       # Dropping the 'failure' column from test_df to get the features and converting to array for model testing
  
    # 3. Initialize the Random Forest Regressor model
    regressor1 = RandomForestRegressor (criterion='squared_error', 
                                        n_estimators=args.n_estimators, max_depth=args.max_depth, 
                                        random_state=42)
  
    # 4. train model
    regressor1 = regressor1.fit(X_train, y_train)
    print(f"model name:   {type(regressor1).__name__}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"max_depth:    {args.max_depth}")

    mlflow.log_param("model",        type(regressor1).__name__)
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth",    args.max_depth)

    # 5. calculate model MSE scores
    y_pred_train = regressor1.predict(X_train)
    y_pred_test  = regressor1.predict(X_test)

    mse_score_train = mean_squared_error(y_train, y_pred_train)
    mse_score_test  = mean_squared_error(y_test,  y_pred_test)
    print(f"Mean Squared Error (Train Dataset): {mse_score_train:.4f}")
    print(f"Mean Squared Error (Test Dataset):  {mse_score_test:.4f}")

    # 6. log model scores
    mlflow.log_metric("MSE Train Score", mse_score_train)
    mlflow.log_metric("MSE Test Score",  mse_score_test)

    # 7. Output the trained model
    mlflow.sklearn.save_model(regressor1, args.model_output)    # args.model_output: must NOT exist, mflow will create one
    #mlflow.sklearn.save_model(regressor1, model_path) 

    mlflow.end_run()                                    # Ending the MLflow experiment run

if __name__ == "__main__":
    main()

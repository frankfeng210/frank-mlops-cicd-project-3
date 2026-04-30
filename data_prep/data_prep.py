
import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",             type=str,                help="Path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="test/train split ratio")
    parser.add_argument("--train_data",       type=str,                help="Path to save train data")
    parser.add_argument("--test_data",        type=str,                help="Path to save test data")
    args = parser.parse_args()

    mlflow.start_run()              # Start MLflow Run

    # Log arguments
    logging.info(f"Input data path:  {args.data}")
    logging.info(f"Test-train ratio: {args.test_train_ratio}")

    # 1. Read source data
    df = pd.read_csv(args.data)

    # 2. Encoding the categorical 'Segment' column
    label_encoder = LabelEncoder()
    df['Segment'] = label_encoder.fit_transform(df['Segment'])

    # Log the first few rows of the dataframe
    logging.info(f"Transformed Data:\n{df.head()}")

    # 3. Split data
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # 4. Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data,  exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "train_data.csv"), index=False)
    test_df.to_csv (os.path.join(args.test_data,  "test_data.csv"), index=False)

    # Log completion
    mlflow.log_metric('train data size', train_df.shape[0])      # Log the train dataset size
    mlflow.log_metric('test data size',  test_df.shape[0])       # Log the test dataset size

    mlflow.end_run()

if __name__ == "__main__":
    main()

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"

def extract_target(
    file_name: str,
    target_col: str,
    logger
):
    train_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}_train_processed.parquet"))
    test_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}_test_processed.parquet"))
    X_train, y_train = train_df.drop(target_col, axis=1), train_df[target_col]
    X_test, y_test = test_df.drop(target_col, axis=1), test_df[target_col]
    logger.info("Fitting the encoder/decoder of target variable")
    logger.info(f"Number of classes: {len(y_train.unique())}")

    return X_train, y_train, X_test, y_test

def trainer(X_train, y_train, logger) -> None:
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_PATH, "random_forest_model.pkl"))
    logger.info("model trained and saved successfully")
    return model
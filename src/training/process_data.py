import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

# Define paths
SOURCE = os.path.join("data", "raw")
DESTINATION = os.path.join("data", "processed")
PREPROCESS_DIR = os.path.join("models", "Preprocessor")


def read_and_process_data(
    file_name: str,
    id_col: str,
    target_col: str,
    logger,
) -> None:
    logger.info("Starting data processing...")

    # Load dataset
    file_path = os.path.join(SOURCE, f"{file_name}.csv")
    df = pd.read_csv(file_path)
    df.set_index(id_col, inplace=True)

    # Define feature columns
    categorical_columns = ['Sex', 'Embarked']
    numerical_columns = ['Age', 'Fare']

    # Drop unnecessary columns
    df.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)

    # Handle missing values
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)

    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df[target_col]
    )

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    # Fit on training data
    preprocessor.fit(train_df)

    # Transform both train and test data
    train_processed = preprocessor.transform(train_df)
    test_processed = preprocessor.transform(test_df)

    # Save preprocessor
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(PREPROCESS_DIR, f"{file_name}_preprocessor.pkl"))

    # Convert to DataFrames with feature names
    feature_names = preprocessor.get_feature_names_out()
    train_processed_df = pd.DataFrame(train_processed, columns=feature_names, index=train_df.index)
    test_processed_df = pd.DataFrame(test_processed, columns=feature_names, index=test_df.index)

    # Add target column back
    train_processed_df[target_col] = train_df[target_col].values
    test_processed_df[target_col] = test_df[target_col].values

    # Save processed data
    os.makedirs(DESTINATION, exist_ok=True)
    train_path = os.path.join(DESTINATION, f"{file_name}_train_processed.parquet")
    test_path = os.path.join(DESTINATION, f"{file_name}_test_processed.parquet")

    train_processed_df.to_parquet(train_path, engine="pyarrow")
    test_processed_df.to_parquet(test_path, engine="pyarrow")

    logger.info("Data processing completed and saved successfully.")

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def read_process_data(
    config,
    file_name: str,
    id_col: str,
    target_col: str,
    logger,
) -> None:
    logger.info("Data Processing started")
    SOURCE = config.data.raw_dir
    DESTINATION = config.data.processed_dir

    df = pd.read_csv(os.path.join(SOURCE, f"{file_name}.csv")) 
    df["Pclass"] = df["Pclass"].astype(str)
    
    X = df.drop([target_col, id_col, "Name", "Ticket", "Cabin"], axis=1)
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train_proc = pipeline.fit_transform(X_train)
    X_val_proc = pipeline.transform(X_val)

    train_df = pd.DataFrame(X_train_proc)
    train_df[target_col] = y_train.values

    val_df = pd.DataFrame(X_val_proc)
    val_df[target_col] = y_val.values

    train_df.to_parquet(
        os.path.join(DESTINATION, f"{file_name}-train.parquet")
    )
    val_df.to_parquet(
        os.path.join(DESTINATION, f"{file_name}-test.parquet"), engine="pyarrow"
    )


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     read_process_data("train", "PassengerId", "Survived", logger)

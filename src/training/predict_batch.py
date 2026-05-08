import duckdb
import pickle
import pandas as pd
from prefect import flow, task
import dagshub
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os

load_dotenv(override=True)

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")


@task
def extract_data():
    con = duckdb.connect(f"md:titanic_db?motherduck_token={MOTHERDUCK_TOKEN}")
    df = con.execute("SELECT * FROM test_data").df()
    con.close()
    return df


@task
def transform_data(df: pd.DataFrame):
    pipeline_path = "data/processed/preprocessing_pipeline.pkl"
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    passenger_ids = df["PassengerId"]
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df["Pclass"] = df["Pclass"].astype(str)

    df_processed = pipeline.transform(df)
    return df_processed, passenger_ids


@task
def predict(df_processed):
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name="MLOps_labs",
        mlflow=True,
    )
    model = mlflow.sklearn.load_model("models:/final_xgb_model@production")
    preds = model.predict(df_processed)
    return preds


@task
def save_predictions(passenger_ids, preds):
    results_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds
    })

    con = duckdb.connect(f"md:titanic_db?motherduck_token={MOTHERDUCK_TOKEN}")
    con.execute("""
        CREATE OR REPLACE TABLE predictions AS
        SELECT * FROM results_df
    """)
    count = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    con.close()

    print(f"✅ Saved {count} predictions to MotherDuck!")
    return results_df


@flow(name="titanic-batch-prediction")
def titanic_batch_flow():
    df = extract_data()
    df_processed, passenger_ids = transform_data(df)
    preds = predict(df_processed)
    save_predictions(passenger_ids, preds)


if __name__ == "__main__":
    titanic_batch_flow()
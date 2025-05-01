import json
import os
import pickle
import joblib # type: ignore


from sklearn.metrics import classification_report # type: ignore


MODEL_PATH = "models"
REPORT_PATH = "reports"

def evaluate(X_test, y_test, model_path):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
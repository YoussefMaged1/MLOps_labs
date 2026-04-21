import json
import os
import pickle
import logging
from skore import EstimatorReport

from train import encode_target_col
MODEL_PATH = "models"
REPORT_PATH = "reports"


def evaluate(X_test, y_test, model_name: str, logger) -> None:
    logger.info(f"Starting evaluation for folder: {model_name}")
    
    translator_path = os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl")
    with open(translator_path, "rb") as pkl:
        translator = pickle.load(pkl)
    y_test_enc = y_test.apply(lambda x: translator["encoder"][x])

    model_files = ["final_xgb_model.pkl", "final_cat_model.pkl"]
    
    all_reports = {}

    for file_name in model_files:
        full_model_path = os.path.join(MODEL_PATH, model_name, file_name)
        
        if not os.path.exists(full_model_path):
            logger.warning(f"Model file {file_name} not found, skipping...")
            continue

        logger.info(f"Evaluating {file_name}...")
        
        with open(full_model_path, "rb") as pkl:
            model = pickle.load(pkl)

        report = EstimatorReport(model, X_test=X_test, y_test=y_test_enc)
        
        model_key = file_name.replace(".pkl", "") # عشان يبقى الاسم نضيف في الـ JSON
        all_reports[model_key] = {
            "estimator_name": report.estimator_name_,
            "metrics": {
                "accuracy": report.metrics.accuracy(),
                "precision": report.metrics.precision(),
                "recall": report.metrics.recall()
            },
            "timings": report.metrics.timings()
        }

    save_dir = os.path.join(REPORT_PATH, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    report_file = os.path.join(save_dir, "evaluation_report.json")
    with open(report_file, "w") as js:
        json.dump(all_reports, js, indent=4)
        
    logger.info(f"Combined evaluation report saved at {report_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    X, y, X_test, y_test = encode_target_col("train", "Survived", "xgb", "cat", logger)
    evaluate(X_test, y_test, "xgb-cat", logger)
import os
import zipfile
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

RAW_DATA_DIR = os.path.join("data", "raw")

def download_titanic(logger) -> str:
    logger.info("Initializing Kaggle API and authenticating...")
    
    api = KaggleApi()
    api.authenticate()

    competition_name = 'titanic'
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    logger.info(f"Downloading files for competition: {competition_name}")
    api.competition_download_files(competition_name, path=RAW_DATA_DIR)

    zip_file_path = os.path.join(RAW_DATA_DIR, f'{competition_name}.zip')
    test_file_path = os.path.join(RAW_DATA_DIR, 'test.csv')
    sub_file_path = os.path.join(RAW_DATA_DIR, 'gender_submission.csv')
    if os.path.exists(zip_file_path):
        logger.info(f"Extracting {zip_file_path}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        logger.info(f"Files extracted successfully to {RAW_DATA_DIR}")
        os.remove(zip_file_path)
        os.remove(test_file_path)
        os.remove(sub_file_path)
    else:
        logger.error("Download failed or zip file not found!")

    return RAW_DATA_DIR

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    download_titanic(logger)
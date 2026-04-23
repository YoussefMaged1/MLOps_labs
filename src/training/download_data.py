import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle

def download_titanic(config, logger) -> str:
    logger.info("Initializing Kaggle API and authenticating...")

    api = KaggleApi()
    api.authenticate()

    competition_name = config.data.competition 
    raw_dir = config.data.raw_dir 
    
    os.makedirs(raw_dir, exist_ok=True)
    
    logger.info(f"Downloading files for: {competition_name}")
    api.competition_download_files(competition_name, path=raw_dir)

    zip_file_path = os.path.join(raw_dir, f"{competition_name}.zip")
    
    if os.path.exists(zip_file_path):
        logger.info(f"Extracting {zip_file_path}...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        
        logger.info(f"Files extracted successfully to {raw_dir}")
        
        os.remove(zip_file_path)
    else:
        logger.error(f"Download failed! Zip file not found at {zip_file_path}")

    return raw_dir

if __name__ == "__main__":
    import logging
    from omegaconf import OmegaConf
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    conf = OmegaConf.create({"data": {"competition": "titanic", "raw_dir": "data/raw"}})
    download_titanic(conf, logger)
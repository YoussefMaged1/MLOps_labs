import os
import shutil
import kagglehub
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
import logging

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def download_titanic(cfg: DictConfig):
    # إعداد الـ logger
    logger = logging.getLogger(__name__)
    
    # تحميل بيانات Kaggle
    download_titanic_data(cfg, logger)

def download_titanic_data(cfg: DictConfig, logger) -> str:
    logger.info(f"Downloading {cfg.data.competition_name} dataset from Kaggle...")
    load_dotenv()

    # الـ Credentials من ملف الـ .env
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN")

    # تحميل الداتا لمكان الـ cache
    path = kagglehub.competition_download(cfg.data.competition_name)

    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        csv_files = files

    # استخدام raw_dir من الـ YAML
    raw_data_dir = cfg.data.raw_dir
    os.makedirs(raw_data_dir, exist_ok=True)

    for csv_file in csv_files:
        source_file = os.path.join(path, csv_file)
        # المسار النهائي: data/raw/train.csv مثلاً
        destination = os.path.join(raw_data_dir, csv_file)
        shutil.copy(source_file, destination)
        logger.info(f"Copied {csv_file} to {destination}")

    logger.info(f"Dataset downloaded and copied to {raw_data_dir}")
    return raw_data_dir

if __name__ == "__main__":
    download_titanic()
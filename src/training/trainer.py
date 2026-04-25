import logging
import hydra
from omegaconf import DictConfig
from src.training.train import encode_target_col, trainer

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../../../conf", config_name="config")
    def main(cfg: DictConfig):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        X, y, X_test, y_test = encode_target_col("train", "Survived", "xgb", "cat", logger)
        trainer(cfg, X, y, "xgb-cat", logger)

    main()
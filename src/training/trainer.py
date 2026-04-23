from src.logger import ExecutorLogger
from src.training.download_data import download_titanic
from src.training.evaluate import evaluate
from src.training.process_data import read_process_data
from src.training.train import encode_target_col, trainer


def main(logger) -> None:
    logger.info("Training started")
    download_titanic(logger)
    read_process_data("train", "PassengerId", "Survived", logger)
    X, y, X_test, y_test = encode_target_col("train", "Survived", "xgb", "cat", logger)
    trainer(X, y, "xgb-cat", logger)
    evaluate(X_test, y_test, "xgb-cat", logger)
    logger.info("Training finished")


if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)

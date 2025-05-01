from src.logger import ExecutorLogger
from src.training.model_evaluate import evaluate
from src.training.data_training import extract_target, trainer
from src.training.process_data import read_and_process_data


def main(logger) -> None:
    logger.info("Training started")
    read_and_process_data("Titanic", "PassengerId", "Survived", logger)
    X, y, X_test, y_test = extract_target("Titanic", "Survived", logger)
    trainer(X, y,logger)
    evaluate(X_test, y_test, "models/random_forest_model.pkl")
    logger.info("Training finished")


if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)

#https://github.com/AyaYasser1112002/Not-Configured_MLOps.git
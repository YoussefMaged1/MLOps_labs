from functools import partial
import logging
import os
import pickle
from typing import Any, Dict

from catboost import CatBoostClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"
N_FOLDS = 5
MAX_EVALS = 4
SPACE = {
    "xgb": {
        "n_estimators": hp.choice("xgb_n_estimators", [100, 200, 300, 500, 1000]),
        "learning_rate": hp.choice("xgb_lr", [0.01, 0.05, 0.1]),
        "max_depth": hp.choice("xgb_max_depth", [3, 5, 7, 10]),
        "subsample": hp.choice("xgb_subsample", [0.8, 1.0]),
    },
    "cat": {
        "depth": hp.choice("cat_depth", [4, 6, 8, 10]),
        "learning_rate": hp.choice("cat_lr", [0.01, 0.05, 0.1]),
        "iterations": hp.choice("cat_iterations", [500, 1000]),
        "l2_leaf_reg": hp.choice("cat_l2", [1, 3, 5, 7]),
    },
}


def encode_target_col(
    file_name: str,
    target_col: str,
    model_1_name: str,
    model_2_name: str,
    logger,
):
    train_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-train.parquet"))
    test_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-test.parquet"))
    X_train, y_train = train_df.drop(target_col, axis=1), train_df[target_col]
    X_test, y_test = test_df.drop(target_col, axis=1), test_df[target_col]
    logger.info("Fitting the encoder/decoder of target variable")
    logger.info(f"Number of classes: {len(y_train.unique())}")
    encoder = {class_: idx for idx, class_ in enumerate(y_train.unique())}
    decoder = {idx: class_ for class_, idx in encoder.items()}
    # save the encoder/decoder of target
    label_translator = {"encoder": encoder, "decoder": decoder}
    logger.info("encoder/decoder of target created successfully")
    if not os.path.exists(os.path.join(MODEL_PATH, f"{model_1_name}-{model_2_name}")):
        os.makedirs(os.path.join(MODEL_PATH, f"{model_1_name}-{model_2_name}"))
    with open(
        os.path.join(
            MODEL_PATH, f"{model_1_name}-{model_2_name}", "model_target_translator.pkl"
        ),
        "wb",
    ) as pkl:
        pickle.dump(label_translator, pkl)
    logger.info("encoder/decoder of target saved")
    return X_train, y_train, X_test, y_test


def objective(params: Dict[str, Any], X, y, n_folds: int = N_FOLDS) -> Dict[str, Any]:

    model1 = XGBClassifier(**params["xgb"], random_state=42)
    model2 = CatBoostClassifier(**params["cat"], random_state=42, verbose=0)
    scores1 = cross_validate(model1, X, y, cv=n_folds, n_jobs=-1, scoring="accuracy")
    score1 = np.mean(scores1["test_score"])

    scores2 = cross_validate(model2, X, y, cv=n_folds, n_jobs=-1, scoring="accuracy")
    score2 = np.mean(scores2["test_score"])

    avg_loss = 1 - ((score1 + score2) / 2)

    return {
        "loss": avg_loss,
        "status": STATUS_OK,
        "params_xgb": params["xgb"],
        "params_cat": params["cat"],
        "score_xgb": score1,
        "score_cat": score2,
    }


def trainer(X, y, model_name: str, logger) -> None:
    logger.info("Load encoder/decoder of target variable")
    with open(
        os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"),
        "rb",
    ) as pkl:
        translator = pickle.load(pkl)
    y_train_enc = y.apply(lambda x: translator["encoder"][x])
    bayes_trials = Trials()
    fmin_objective = partial(objective, X=X, y=y_train_enc)
    logger.info("optimization started")
    fmin(
        fn=fmin_objective,
        space=SPACE,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=bayes_trials,
    )
    logger.info("optimization completed")

    best_trial = bayes_trials.results[
        np.argmin([r["loss"] for r in bayes_trials.results])
    ]

    best_params_xgb = best_trial["params_xgb"]
    best_params_cat = best_trial["params_cat"]

    logger.info("Training final models with best found parameters")

    final_xgb = XGBClassifier(**best_params_xgb, random_state=42)
    final_xgb.fit(X, y_train_enc)

    final_cat = CatBoostClassifier(**best_params_cat, random_state=42, verbose=0)
    final_cat.fit(X, y_train_enc)

    save_path = os.path.join(MODEL_PATH, model_name)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "final_xgb_model.pkl"), "wb") as pkl:
        pickle.dump(final_xgb, pkl)

    with open(os.path.join(save_path, "final_cat_model.pkl"), "wb") as pkl:
        pickle.dump(final_cat, pkl)

    logger.info(
        f"Both models (XGB & CatBoost) trained and saved successfully in {save_path}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    X, y, X_test, y_test = encode_target_col("train", "Survived", "xgb", "cat", logger)
    trainer(X, y, "xgb-cat", logger)

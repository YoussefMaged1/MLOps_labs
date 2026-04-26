import logging
import os
import pickle
from functools import partial
from typing import Any, Dict

import dagshub
import mlflow
import mlflow.catboost
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"
N_FOLDS = 5

def get_space(hyper_cfg):
    return {
        'xgb': {
            'n_estimators': hp.choice('xgb_n_estimators', list(hyper_cfg.xgb.n_estimators)),
            'learning_rate': hp.choice('xgb_lr', list(hyper_cfg.xgb.learning_rate)),
            'max_depth': hp.choice('xgb_max_depth', list(hyper_cfg.xgb.max_depth)),
            'subsample': hp.choice('xgb_subsample', list(hyper_cfg.xgb.subsample)),
        },
        'cat': {
            'depth': hp.choice('cat_depth', list(hyper_cfg.cat.depth)),
            'learning_rate': hp.choice('cat_lr', list(hyper_cfg.cat.learning_rate)),
            'iterations': hp.choice('cat_iterations', list(hyper_cfg.cat.iterations)),
            'l2_leaf_reg': hp.choice('cat_l2', list(hyper_cfg.cat.l2_leaf_reg)),
        }
    }

def encode_target_col(file_name, target_col, model_1_name, model_2_name, logger):
    train_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-train.parquet"))
    test_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-test.parquet"))
    X_train, y_train = train_df.drop(target_col, axis=1), train_df[target_col]
    X_test, y_test = test_df.drop(target_col, axis=1), test_df[target_col]
    
    encoder = {class_: idx for idx, class_ in enumerate(y_train.unique())}
    decoder = {idx: class_ for class_, idx in encoder.items()}
    label_translator = {"encoder": encoder, "decoder": decoder}
    
    save_dir = os.path.join(MODEL_PATH, f"{model_1_name}-{model_2_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "model_target_translator.pkl"), "wb") as pkl:
        pickle.dump(label_translator, pkl)
    
    return X_train, y_train, X_test, y_test

def objective(params: Dict[str, Any], X, y, n_folds: int = N_FOLDS) -> Dict[str, Any]:
    # فتح Nested Run لكل تجربة في الـ Optimization
    with mlflow.start_run(nested=True):
        model1 = XGBClassifier(**params["xgb"], random_state=42)
        model2 = CatBoostClassifier(**params["cat"], random_state=42, verbose=0)
        
        scores1 = cross_validate(model1, X, y, cv=n_folds, n_jobs=-1, scoring="accuracy")
        score1 = np.mean(scores1["test_score"])

        scores2 = cross_validate(model2, X, y, cv=n_folds, n_jobs=-1, scoring="accuracy")
        score2 = np.mean(scores2["test_score"])

        avg_loss = 1 - ((score1 + score2) / 2)

        # تسجيل الـ Parameters والـ Metrics لكل محاولة
        mlflow.log_params({f"xgb_{k}": int(v) if isinstance(v, (int, float)) and v == int(v) else round(float(v), 4) if isinstance(v, float) else str(v) for k, v in params["xgb"].items()})
        mlflow.log_params({f"cat_{k}": int(v) if isinstance(v, (int, float)) and v == int(v) else round(float(v), 4) if isinstance(v, float) else str(v) for k, v in params["cat"].items()})
        mlflow.log_metric("xgb_accuracy", score1)
        mlflow.log_metric("cat_accuracy", score2)
        mlflow.log_metric("avg_loss", avg_loss)

        return {
            "loss": avg_loss,
            "status": STATUS_OK,
            "params_xgb": params["xgb"],
            "params_cat": params["cat"],
        }

def trainer(cfg, X, y, model_name: str, logger) -> None:
    logger.info("Load encoder/decoder of target variable")
    with open(os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"), "rb") as pkl:
        translator = pickle.load(pkl)
    
    y_train_enc = y.apply(lambda x: translator["encoder"][x])
    
    with mlflow.start_run(run_name="Hyperparameter_Optimization"):
        bayes_trials = Trials()
        fmin_objective = partial(objective, X=X, y=y_train_enc)
        
        logger.info("Optimization started")
        space = get_space(cfg.hyperparams) 

        fmin(
            fn=fmin_objective,
            space=space,  
            algo=tpe.suggest,
            max_evals=cfg.hyperparams.max_evals, 
            trials=bayes_trials,
        )
        
        # الحصول على أفضل النتائج
        best_trial = bayes_trials.results[np.argmin([r["loss"] for r in bayes_trials.results])]
        best_params_xgb = best_trial["params_xgb"]
        best_params_cat = best_trial["params_cat"]

        logger.info("Training final models with best found parameters")
        
        final_xgb = XGBClassifier(**best_params_xgb, random_state=42)
        final_xgb.fit(X, y_train_enc)
        
        final_cat = CatBoostClassifier(**best_params_cat, random_state=42, verbose=0)
        final_cat.fit(X, y_train_enc)

        mlflow.sklearn.log_model(final_xgb, "final_xgb_model")
        mlflow.catboost.log_model(final_cat, "final_cat_model")
        
        save_path = os.path.join(MODEL_PATH, model_name)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "final_xgb_model.pkl"), "wb") as pkl:
            pickle.dump(final_xgb, pkl)
        with open(os.path.join(save_path, "final_cat_model.pkl"), "wb") as pkl:
            pickle.dump(final_cat, pkl)

        logger.info(f"Models saved and logged to MLflow successfully")

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def main(cfg: DictConfig):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        dagshub.init(repo_owner='youssef.maged237', repo_name='MLOps_labs', mlflow=True)
        X, y, X_test, y_test = encode_target_col("train", "Survived", "xgb", "cat", logger)
        trainer(cfg, X, y, "xgb-cat", logger)

    main()
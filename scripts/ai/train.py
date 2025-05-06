import json
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

# Paths
DATA_PATH = Path("data/featured/loans_featured.parquet")
SAVE_DIR = Path("data/model")
METRICS_DIR = Path("data/metrics")
MLFLOW_DIR = Path("data/mlruns")

SAVE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(str(MLFLOW_DIR))

# Configuration parameters
EARLY_STOPPING_ROUNDS = 50
TIME_SERIES_SPLITS = 3
BASELINE_LEARNING_RATE = 0.05
N_PARALLEL_JOBS = 6
OPTUNA_TRIALS = 50

# Set random seed
SEED = 11
random.seed(SEED)
np.random.seed(SEED)



def identify_categorical_features(X):
    categorical_features = []
    for col in X.columns:
        # Identify potential categorical columns based on column name and unique values
        if any(keyword in col.lower() for keyword in ['grade', 'type', 'status', 'purpose', 'home', 'state', 'term', 'flag']):
            if X[col].nunique() < 50:  # Prevent high cardinality columns
                categorical_features.append(col)
    return categorical_features

def baseline_model(bt, X_tr, y_tr, X_val, y_val):
    print(f"[{datetime.now()}] Training baseline {bt} model...")
    start_time = time.time()
    
    # Identify categorical features
    categorical_features = identify_categorical_features(X_tr)
    print(f"[{datetime.now()}] Using {len(categorical_features)} categorical features: {categorical_features}")
    
    model = lgb.LGBMClassifier(
        boosting_type=bt,
        objective="binary",
        n_estimators=150,
        learning_rate=BASELINE_LEARNING_RATE,
        random_state=SEED,
        is_unbalance=True,
        verbose=-1
    )
    
    # Save results
    evals_result = {}
    
    # Add early stopping
    callbacks = [lgb.record_evaluation(evals_result)]
    if bt != 'dart':  # DART doesn't support early stopping
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS))
    
    model.fit(
        X_tr, y_tr, 
        eval_set=[(X_val, y_val)], 
        eval_metric='auc', 
        categorical_feature=categorical_features,
        callbacks=callbacks
    )
    
    preds = model.predict_proba(X_val)[:, 1]
    print(f"[{datetime.now()}] Baseline {bt} model training completed in {time.time() - start_time:.2f} seconds")
    return model, preds, model.get_params(), evals_result

def cross_val_objective(params, X, y, cv_splits):
    """
    Cross-validation.
    """
    categorical_features = identify_categorical_features(X)
    scores = []
    
    for train_idx, val_idx in cv_splits:
        X_tr_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_tr_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = lgb.LGBMClassifier(**params)
        
        # Add early stopping
        callbacks = []
        if params.get('boosting_type') != 'dart':
            callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS))
            
        model.fit(
            X_tr_fold, 
            y_tr_fold, 
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='auc',
            categorical_feature=categorical_features,
            callbacks=callbacks,
            verbose=False
        )
        
        preds = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, preds)
        scores.append(score)
    
    return np.mean(scores)

def tuned_model(bt, X_tr, y_tr, X_val, y_val):
    print(f"[{datetime.now()}] Training tuned {bt} model with Optuna...")
    start_time = time.time()
    
    # Create cross-validation splits using issue_year
    all_X = pd.concat([X_tr, X_val])
    all_y = pd.concat([y_tr, y_val])
    
    # Create TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLITS)
    cv_splits = list(tscv.split(all_X))
    
    # Define categorical features for the full dataset
    categorical_features = identify_categorical_features(all_X)
    
    def objective(trial):
        params = {
            "boosting_type": bt,
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "random_state": SEED,
            "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_jobs": 1
        }
        
        # If not using is_unbalance, use scale_pos_weight
        if not params["is_unbalance"]:
            params["scale_pos_weight"] = (y_tr == 0).sum() / (y_tr == 1).sum()
            
        return cross_val_objective(params, all_X, all_y, cv_splits)

    # Create and save study
    study = optuna.create_study(direction="maximize", study_name=f"{bt}_study")
    
    # Use parallel processing to speed up hyperparameter search
    print(f"[{datetime.now()}] Starting parallel Optuna optimization with {N_PARALLEL_JOBS} workers...")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=N_PARALLEL_JOBS)
    
    # Save study
    study_path = METRICS_DIR / f"{bt}_study.pkl"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    best = study.best_params
    print(f"[{datetime.now()}] Best Optuna parameters for {bt}: {best}")
    
    # Final train on all data with best params
    best["n_jobs"] = -1
    model = lgb.LGBMClassifier(**best)
    
    # Save evaluation results
    evals_result = {}
    
    # Add early stopping
    callbacks = [lgb.record_evaluation(evals_result)]
    if bt != 'dart':  # DART doesn't support early stopping
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS))
    
    model.fit(
        X_tr, y_tr, 
        eval_set=[(X_val, y_val)], 
        eval_metric='auc', 
        categorical_feature=categorical_features,
        callbacks=callbacks,
        verbose=-1  # Reduce verbosity
    )
    
    preds = model.predict_proba(X_val)[:, 1]
    print(f"[{datetime.now()}] Tuned {bt} model training completed in {time.time() - start_time:.2f} seconds")
    return model, preds, best, evals_result

def log_and_save(model, preds, y_val, params, run_name, file_name, evals_result=None):
    auc_roc = roc_auc_score(y_val, preds)
    auc_pr  = average_precision_score(y_val, preds)
    fpr, tpr, _ = roc_curve(y_val, preds)
    ks = float(max(tpr - fpr))
    
    # Save ROC and PR curve
    model_name = file_name.split('.')[0]
    model_metrics_dir = METRICS_DIR / model_name
    model_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_val, preds)
    np.save(model_metrics_dir / "fpr.npy", fpr)
    np.save(model_metrics_dir / "tpr.npy", tpr)
    
    precision, recall, _ = precision_recall_curve(y_val, preds)
    np.save(model_metrics_dir / "precision.npy", precision)
    np.save(model_metrics_dir / "recall.npy", recall)
    
    # Save learning curves if available
    if evals_result:
        with open(model_metrics_dir / "evals_result.json", 'w') as f:
            json.dump(evals_result, f)

    try:
        with mlflow.start_run(run_name=run_name):
            print(f"[{datetime.now()}] Logging metrics to MLflow for {run_name}...")
            mlflow.log_params(params)
            mlflow.log_metric("val_auc_roc", auc_roc)
            mlflow.log_metric("val_auc_pr",  auc_pr)
            mlflow.log_metric("val_ks",      ks)
            mlflow.lightgbm.log_model(model, artifact_path="model")
            
            # Log curve data to MLflow as well
            with open(model_metrics_dir / "fpr.npy", "rb") as f:
                mlflow.log_artifact(model_metrics_dir / "fpr.npy", "roc")
            with open(model_metrics_dir / "tpr.npy", "rb") as f:
                mlflow.log_artifact(model_metrics_dir / "tpr.npy", "roc")
            with open(model_metrics_dir / "precision.npy", "rb") as f:
                mlflow.log_artifact(model_metrics_dir / "precision.npy", "pr")
            with open(model_metrics_dir / "recall.npy", "rb") as f:
                mlflow.log_artifact(model_metrics_dir / "recall.npy", "pr")
            
            if evals_result:
                with open(model_metrics_dir / "evals_result.json", "r") as f:
                    mlflow.log_artifact(model_metrics_dir / "evals_result.json")
            print(f"[{datetime.now()}] MLflow logging completed for {run_name}")
    except Exception as e:
        print(f"[{datetime.now()}] Error logging to MLflow: {e}")
        print(f"[{datetime.now()}] Continuing without MLflow logging")

    model.booster_.save_model(SAVE_DIR / file_name)

def main():
    print(f"[{datetime.now()}] Starting model training...")
    overall_start_time = time.time()

    # Create MLflow experiment if it doesn't exist
    print(f"[{datetime.now()}] Setting up MLflow experiment...")
    try:
        experiment = mlflow.get_experiment_by_name("credit_default")
        if experiment is None:
            experiment_id = mlflow.create_experiment("credit_default")
            print(f"[{datetime.now()}] Created new MLflow experiment with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"[{datetime.now()}] Using existing MLflow experiment with ID: {experiment_id}")
        
        mlflow.set_experiment("credit_default")
        print(f"[{datetime.now()}] Set active experiment to 'credit_default'")
    except Exception as e:
        print(f"[{datetime.now()}] Error setting up MLflow experiment: {e}")
        print(f"[{datetime.now()}] Will proceed without MLflow tracking")
    
    df = pd.read_parquet(DATA_PATH)

    # Prints counts by year
    print(df["issue_year"].value_counts().sort_index())
    
    # Print class imbalance info
    pos_rate = df["target"].mean()
    print(f"[{datetime.now()}] Target distribution: {pos_rate:.4f} positive rate (1:{(1-pos_rate)/pos_rate:.1f} imbalance)")

    # Define trainin and validation periods
    train_mask = df["issue_year"] <= 2016
    val_mask   = df["issue_year"] == 2017

    X_train, y_train = df[train_mask].drop(columns=["target", "issue_year"]), df[train_mask]["target"]
    X_val,   y_val   = df[val_mask].drop(columns=["target", "issue_year"]),   df[val_mask]["target"]
    
    print(f"[{datetime.now()}] Training data: {X_train.shape[0]} rows, Validation data: {X_val.shape[0]} rows")

    for bt in ["gbdt", "dart", "goss"]:
        # baseline
        model, preds, params, evals_result = baseline_model(bt, X_train, y_train, X_val, y_val)
        log_and_save(model, preds, y_val, params,
                     run_name=f"baseline_{bt}",
                     file_name=f"{bt}_baseline.txt",
                     evals_result=evals_result)

        # tuned with Optuna
        model, preds, params, evals_result = tuned_model(bt, X_train, y_train, X_val, y_val)
        log_and_save(model, preds, y_val, params,
                     run_name=f"tuned_{bt}",
                     file_name=f"{bt}_tuned.txt",
                     evals_result=evals_result)

    print(f"[{datetime.now()}] All model training completed in {time.time() - overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

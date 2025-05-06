import os
import time
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)

DATA_PATH = Path("data/featured/loans_featured.parquet")
MODEL_DIR = Path("data/model")
METRICS_DIR = Path("data/metrics")
OUTPUT = METRICS_DIR / "summary.parquet"

def compute_ks(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(max(tpr - fpr))

def load_models():
    return sorted(MODEL_DIR.glob("*.txt"))

def evaluate_model(model_path, X_test, y_test):
    print(f"[{datetime.now()}] Evaluating model: {model_path.name}")
    start_time = time.time()
    booster = lgb.Booster(model_file=str(model_path))
    preds   = booster.predict(X_test)
    auc_roc = roc_auc_score(y_test, preds)
    auc_pr  = average_precision_score(y_test, preds)
    ks      = compute_ks(y_test, preds)
    
    # confusion at 0.5
    y_pred_label = (preds >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_label).ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    f1 = f1_score(y_test, y_pred_label)
    brier = brier_score_loss(y_test, preds)
    
    # Calculate model size
    model_size = os.path.getsize(model_path) / 1024
    
    # Save ROC and PR curve data for dashboard
    model_name = model_path.name.split('.')[0]
    model_metrics_dir = METRICS_DIR / model_name
    model_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ROC curve data
    fpr, tpr, _ = roc_curve(y_test, preds)
    np.save(model_metrics_dir / "fpr.npy", fpr)
    np.save(model_metrics_dir / "tpr.npy", tpr)
    
    # Save PR curve data
    precision, recall, _ = precision_recall_curve(y_test, preds)
    np.save(model_metrics_dir / "precision.npy", precision)
    np.save(model_metrics_dir / "recall.npy", recall)
    
    eval_time = time.time() - start_time
    print(f"[{datetime.now()}] Evaluation of {model_path.name} completed in {eval_time:.2f} seconds")
    print(f"[{datetime.now()}] Metrics - AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}, KS: {ks:.4f}, F1: {f1:.4f}")
    
    return {
        "model": model_path.name,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "ks": ks,
        "accuracy": accuracy,
        "f1_score": f1,
        "brier_score": brier,
        "latency_or_model_size": model_size,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

def main():
    print(f"[{datetime.now()}] Starting model evaluation...")
    overall_start_time = time.time()
    df = pd.read_parquet(DATA_PATH)

    test_mask = df["issue_year"] == 2018
    X_test = df[test_mask].drop(columns=["target", "issue_year"])
    y_test = df[test_mask]["target"]

    records = []
    for model_path in load_models():
        rec = evaluate_model(model_path, X_test, y_test)
        records.append(rec)
        print(f"Evaluated {model_path.name}: AUC-ROC={rec['auc_roc']:.3f}, AUC-PR={rec['auc_pr']:.3f}, KS={rec['ks']:.3f}")

    # save summary
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(OUTPUT, index=False)
    
    total_time = time.time() - overall_start_time
    print(f"[{datetime.now()}] Model evaluation completed in {total_time:.2f} seconds")
    print(f"[{datetime.now()}] Saved all metrics to {OUTPUT}")

if __name__ == "__main__":
    main()

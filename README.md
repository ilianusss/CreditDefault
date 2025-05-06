# Credit‑Default Prediction 📊💳

End-to-end machine learning pipeline to predict loan defaults, including data cleaning, feature engineering, training of six models, and ranking via a dashboard.

| Stage           | Tech                       |
|-----------------|----------------------------|
| **Storage**     | PostgreSQL 16 · Parquet    |
| **ML & Tuning** | LightGBM · Optuna · MLflow |
| **Viz / UI**    | Streamlit · Plotly         |
| **Language**    | Python 3.12                |

---

## 1 Project Overview
* **Dataset** – Lending Club *accepted* loans 2007 – 2018 Q4 (`accepted_2007_to_2018Q4.csv.gz`) → 1 345 310 rows after cleaning.  
* **Data processing**  
  1. **prepare_data.py** – loads the raw *.gz, cleans & inserts into `curated.loans_clean`.  
  2. **features.py** – engineers 85 features (financial ratios, temporal lags, one‑hot grades, FICO ranges) and exports `data/featured/loans_featured.parquet`.  
* **Modeling**  
  * Six LightGBM models (gbdt / dart / goss × baseline & Optuna‑tuned).  
  * Out‑of‑time splits : train ≤ 2016, validation 2017, test 2018.  
  * Metrics logged in MLflow ; best model ≈ AUC‑ROC 0.72, KS 0.46.  
* **Dashboard**  
  * Leaderboard ranks models by weighted score (AUC‑ROC 0.35, KS 0.25, AUC‑PR 0.20, F1 0.10, Brier 0.05, Latency/Size 0.05).  
  * Tabs show ROC/PR curves, learning curve, and confusion matrix.

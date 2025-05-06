import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.data.db_connect import get_db_connection

def emp_length_num(x):
    if pd.isna(x):
        return np.nan
    if "< 1" in x:
        return 0.5
    if "10+" in x:
        return 10
    try:
        return int(x.split()[0])
    except:
        return np.nan

def financial_ratios(df):
    df["dti_ratio"]        = df["installment"] / (df["annual_inc"] / 12)
    term_months            = df["term"].str.extract(r"(\d+)").astype(float)[0]
    df["payment_to_loan"]  = df["installment"] * term_months / df["loan_amnt"]
    df["installment_ratio"]= df["installment"] / df["annual_inc"]
    return df

def temporal_features(df):
    # Ensure issue_d is a datetime64 type
    if not pd.api.types.is_datetime64_dtype(df["issue_d"]):
        df["issue_d"] = pd.to_datetime(df["issue_d"])
    
    ref_date = pd.Timestamp("2018-12-31")
    df["loan_age_days"]  = (ref_date - df["issue_d"]).dt.days
    df["issue_month"]    = df["issue_d"].dt.month
    df["issue_quarter"]  = df["issue_d"].dt.quarter
    return df

def encode_str(df):
    """
    One-hot encode string columns
    """
    cols = [
        "term", "grade", "sub_grade",
        "home_ownership", "verification_status", "purpose"
    ]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    for col in cols:
        if col in df.columns:
            arr = encoder.fit_transform(df[[col]])
            names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            df = pd.concat(
                [df, pd.DataFrame(arr, columns=names, index=df.index)],
                axis=1
            )
            df.drop(columns=[col], inplace=True)
    return df

def main() -> None:
    print(f"[{datetime.now()}] Starting feature engineering...")
    start_time = time.time()
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        print(f"[{datetime.now()}] Reading data from database...")
        df = pd.read_sql("SELECT * FROM curated.loans_clean", conn)
        conn.close()
        print(f"[{datetime.now()}] Read {len(df)} rows from database.")
    except Exception as e:
        print(f"Error reading data from database: {e}")
        return

    # Calculating features
    print(f"[{datetime.now()}] Calculating financial ratios...")
    df = financial_ratios(df)
    
    print(f"[{datetime.now()}] Calculating temporal features...")
    df = temporal_features(df)
    
    print(f"[{datetime.now()}] Processing employment length...")
    df["emp_length_num"] = df["emp_length"].map(emp_length_num)

    print(f"[{datetime.now()}] Encoding categorical features...")
    df = encode_str(df)

    # Select the data we want to use for the training
    raw_numeric = [
        "loan_amnt", "int_rate", "annual_inc", "dti",
        "open_acc", "total_acc", "installment",
        "funded_amnt", "funded_amnt_inv",
        "revol_util"
    ]
    
    derived = [
        "dti_ratio", "payment_to_loan", "installment_ratio",
        "loan_age_days", "issue_month", "issue_quarter",
        "emp_length_num"
    ]

    oh_prefixes = ["term_", "grade_", "sub_grade_", "home_ownership_",
                   "verification_status_", "purpose_"]
    one_hot = [col for col in df.columns
               for pre in oh_prefixes if col.startswith(pre)]

    # Saving in a praquet file
    final_cols = ["target"] + raw_numeric + derived + one_hot
    final = df[final_cols]

    out_dir = Path("data/featured")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"loans_featured.parquet"
    print(f"[{datetime.now()}] Saving featured data to {out_path}...")
    final.to_parquet(out_path, index=False)
    
    total_time = time.time() - start_time
    print(f"[{datetime.now()}] Feature engineering completed in {total_time:.2f} seconds. Processed {len(df)} rows with {len(final.columns)} features.")

if __name__ == "__main__":
    main()

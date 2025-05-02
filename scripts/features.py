import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import psycopg2
from sklearn.preprocessing import OneHotEncoder

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
    ref_date = pd.Timestamp("2018-12-31")
    df["loan_age_days"]  = (ref_date - df["issue_d"]).dt.days
    df["issue_month"]    = df["issue_d"].dt.month
    df["issue_quarter"]  = df["issue_d"].dt.quarter
    return df

def encode_str(df):
    #We have to encode the strings since the AI can only process values.
    cat_cols = [
        "term", "grade", "sub_grade",
        "home_ownership", "verification_status", "purpose"
    ]
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    for col in cat_cols:
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
    conn = psycopg2.connect(
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASS"),
        host=os.getenv("PG_HOST")
    )
    df = pd.read_sql("SELECT * FROM curated.loans_clean", conn)
    conn.close()

    df = financial_ratios(df)
    df = temporal_features(df)
    df["emp_length_num"] = df["emp_length"].map(emp_length_num)

    df = encode_str(df)

    # Select the data we want to use for the AI
    raw_numeric = [
        "loan_amnt", "int_rate", "annual_inc", "dti",
        "open_acc", "total_acc", "installment",
        "funded_amnt", "funded_amnt_inv",
        "acc_now_delinq", "delinq_2yrs",
        "fico_range_low", "fico_range_high"
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

    final_cols = ["target"] + raw_numeric + derived + one_hot
    final = df[final_cols]

    out_dir = Path("data/featured")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"loans_features_{datetime.now():%Y%m%d}.parquet"
    final.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()

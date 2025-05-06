import gzip
import hashlib
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.data.db_connect import get_db_connection

def main() -> None:
    """
    Clean the data from the csv.gz and load it into the db.
    """
    print(f"[{datetime.now()}] Starting data preparation...")
    conn = get_db_connection()
    if conn is None:
        return

    # Compute checksum
    file_path = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
    with file_path.open("rb") as f_bin:
        checksum = hashlib.md5(f_bin.read()).hexdigest()
    
    # Basic cleaning
    print(f"[{datetime.now()}] Reading CSV file...")
    start_time = time.time()
    with gzip.open(file_path, "rt") as f_txt:
        df = pd.read_csv(f_txt, low_memory=False)
    print(f"[{datetime.now()}] CSV file read in {time.time() - start_time:.2f} seconds")
    
    print(f"[{datetime.now()}] Cleaning data...")
    df = df.dropna(how="all")  # Deletes all empty lines 
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]

    # Conversions
    print(f"[{datetime.now()}] Converting data types...")
    
    # Check int rate is already numeric or needs string conversion
    #if pd.api.types.is_string_dtype(df["int_rate"]):
    #    df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)
    #else:
    #    # Already numeric, no conversion needed
    #    print(f"[{datetime.now()}] int_rate is already numeric, skipping conversion")
    
    # Handle revol_util - check if it's already numeric or needs string conversion
    #if pd.api.types.is_string_dtype(df["revol_util"]):
    #    df["revol_util"] = df["revol_util"].str.rstrip("%").astype(float)
    #else:
    #    # Already numeric, no conversion needed
    #    print(f"[{datetime.now()}] revol_util is already numeric, skipping conversion")
    
    df["issue_d"]     = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["dti"]         = pd.to_numeric(df["dti"], errors="coerce")
    
    # Convert float columns to integer
    print(f"[{datetime.now()}] Converting to integers...")
    integer_columns = ["open_acc", "total_acc"]
    for col in integer_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Add target column
    df["target"]      = df["loan_status"].map({"Fully Paid": 0, "Charged Off": 1})

    # Select the columns we want
    cols = [
        "target", "loan_amnt", "term", "int_rate", "grade", "sub_grade",
        "emp_length", "home_ownership", "annual_inc", "purpose", "dti",
        "issue_d", "revol_util", "open_acc", "total_acc",
        "installment", "funded_amnt", "funded_amnt_inv",
        "verification_status", "zip_code"
    ]
    df = df[cols]

    # Ingestion logs
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO raw.loans_ingestion(ts, checksum, filename)
            VALUES (%s, %s, %s)
        """, (datetime.now(), checksum, file_path.name))
    conn.commit()

    # Copy cleaned DataFrame into db
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY curated.loans_clean FROM STDIN WITH CSV HEADER",
            buffer
        )
    conn.commit()

    conn.close()
    
    print(f"[{datetime.now()}] Data preparation completed. Processed {len(df)} rows.")

if __name__ == "__main__":
    main()

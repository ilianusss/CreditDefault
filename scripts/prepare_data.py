import os
import gzip
import hashlib
from io import StringIO
from pathlib import Path
from datetime import datetime
import pandas as pd
import psycopg2

def main() -> None:
    """
    Clean the data from the csv.gz and load it into the db.
    """
    # 1. Connect to the DB
    conn = psycopg2.connect(
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASS"),
        host=os.getenv("PG_HOST")
    )

    # 2. Compute checksum
    file_path = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
    with file_path.open("rb") as f_bin:
        checksum = hashlib.md5(f_bin.read()).hexdigest()
    
    # 3. Basic cleaning
    with gzip.open(file_path, "rt") as f_txt:
        df = pd.read_csv(f_txt, low_memory=False)
    
    df = df.dropna(how="all")  # Deletes all empty lines 
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]

    # Conversions
    df["int_rate"]    = df["int_rate"].str.rstrip("%").astype(float)
    df["revol_util"]  = df["revol_util"].str.rstrip("%").astype(float)
    df["issue_d"]     = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["dti"]         = pd.to_numeric(df["dti"], errors="coerce")
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

    # 4. Copy cleaned DataFrame into db 
    # Open the csv in a buffer and set the cursor to the start of it
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Copy into the db
    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY curated.loans_clean FROM STDIN WITH CSV HEADER",
            buffer
        )
    conn.commit()

    conn.close()

if __name__ == "__main__":
    main()

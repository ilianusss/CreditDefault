-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS curated;

-- Create logs tables
CREATE TABLE IF NOT EXISTS raw.loans_ingestion (
    ts        TIMESTAMP NOT NULL,
    checksum  TEXT      NOT NULL,
    filename  TEXT      NOT NULL
);

-- Create loans tables
DROP TABLE IF EXISTS curated.loans_clean;

CREATE TABLE curated.loans_clean (
    target               SMALLINT,
    loan_amnt            REAL,
    term                 TEXT,
    int_rate             REAL,
    grade                TEXT,
    sub_grade            TEXT,
    emp_length           TEXT,
    home_ownership       TEXT,
    annual_inc           REAL,
    purpose              TEXT,
    dti                  REAL,
    issue_d              DATE,
    revol_util           REAL,
    open_acc             INTEGER,
    total_acc            INTEGER,
    installment          REAL,
    funded_amnt          REAL,
    funded_amnt_inv      REAL,
    verification_status  TEXT,
    zip_code             TEXT
);


import os
import psycopg2
from pathlib import Path

def main() -> None:
    """
    Execute db.sql to create schemas and tables.
    """
    # 1. Connect to the DB
    conn = psycopg2.connect(
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASS"),
        host=os.getenv("PG_HOST")
    )

    # 2. Read and execute db.sql
    sql_path = Path("scripts/db.sql")
    # Fancy version sql_path = Path(__file__).parent / "db.sql"
    sql = sql_path.read_text()

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()

import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.data.db_connect import get_db_connection

def main() -> None:
    """
    Execute db.sql to create schemas and tables.
    """
    print(f"[{datetime.now()}] Setting up database...")
    
    conn = get_db_connection()
    if conn is None:
        return  # Connection failed

    sql_path = Path("scripts/data/db.sql")
    sql = sql_path.read_text()

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()

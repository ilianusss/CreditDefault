import os
import subprocess
import psycopg2

def get_system_username():
    """
    Get the system username using whoami.
    """
    try:
        result = subprocess.run(['whoami'], capture_output=True, text=True)
        username = result.stdout.strip()
        return username
    except Exception as e:
        print(f"Error getting system username: {e}")

def get_db_connection():
    """
    Create connection to the db.
    """
    system_username = get_system_username()
    
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PG_DB", "postgres"),
            user=os.getenv("PG_USER", system_username),
            password=os.getenv("PG_PASS", ""),
            host=os.getenv("PG_HOST", "localhost")
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Database connection error: {e}")
        print("Please ensure PostgreSQL is running and environment variables are set correctly.")
        return None

import subprocess
import sys
import os
import time
from datetime import datetime

os.makedirs("data/model", exist_ok=True)
os.makedirs("data/metrics", exist_ok=True)
os.makedirs("data/mlruns", exist_ok=True)
os.makedirs("data/featured", exist_ok=True)

PYTHON_CMD = "python3.10"

def log(message):
    """Log a message with timestamp"""
    print(f"[{datetime.now()}] {message}")

def run_with_retry(cmd, name, retries=1, retry_delay=5):
    """Run a command with retries"""
    for attempt in range(retries + 1):
        try:
            log(f"Running {name}... (Attempt {attempt + 1}/{retries + 1})")
            subprocess.run(cmd, check=True)
            log(f"{name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            log(f"Error in {name}: {e}")
            if attempt < retries:
                log(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                log(f"All {retries + 1} attempts failed for {name}")
                return False

def prepare_data():
    """Run data preparation: create schemas and clean raw data into DB."""
    log("Starting data preparation...")
    if not run_with_retry([PYTHON_CMD, "scripts/data/db.py"], "database schema creation"):
        return False
    if not run_with_retry([PYTHON_CMD, "scripts/data/prepare_data.py"], "data preparation"):
        return False
    return True

def build_features():
    """Generate feature snapshot parquet file."""
    log("Starting feature engineering...")
    return run_with_retry([PYTHON_CMD, "scripts/data/features.py"], "feature engineering")

def train_models():
    """Train baseline and tuned LightGBM models."""
    log("Starting model training...")
    return run_with_retry([PYTHON_CMD, "scripts/ai/train.py"], "model training")

def evaluate_models():
    """Evaluate trained models on test set and save metrics."""
    log("Starting model evaluation...")
    return run_with_retry([PYTHON_CMD, "scripts/ai/evaluate.py"], "model evaluation")

def launch_dashboard():
    """Start Streamlit dashboard."""
    log("Launching dashboard...")
    try:
        # Using python -m streamlit to ensure the correct Python environment
        subprocess.run([PYTHON_CMD, "-m", "streamlit", "run", "scripts/dashboard/dashboard.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        log(f"Error launching dashboard: {e}")
        return False
    except KeyboardInterrupt:
        log("Dashboard stopped by user")
        return True

def orchestrator():
    """End-to-end workflow: prepare data, build features, train, evaluate, and launch dashboard."""
    log("Starting end-to-end pipeline...")
    
    # Run each step and check for errors
    if not prepare_data():
        log("Data preparation failed. Stopping pipeline.")
        return False
    
    if not build_features():
        log("Feature engineering failed. Stopping pipeline.")
        return False
    
    if not train_models():
        log("Model training failed. Stopping pipeline.")
        return False
    
    if not evaluate_models():
        log("Model evaluation failed. Stopping pipeline.")
        return False
        
    # Launch dashboard at the end (this will block until dashboard is closed)
    launch_dashboard()
    
    log("Pipeline completed successfully")
    return True

if __name__ == "__main__":
    success = orchestrator()
    sys.exit(0 if success else 1)

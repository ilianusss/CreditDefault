.PHONY: all data train evaluate dashboard clean help metrics

PYTHON = python3.10

help:
	@echo "Available commands:"
	@echo "  make all        	- Run the complete pipeline from data preparation to dashboard"
	@echo "  make data       	- Run data preparation (db setup and data cleaning)"
	@echo "  make features   	- Run feature engineering"
	@echo "  make train      	- Train models"
	@echo "  make evaluate   	- Evaluate models"
	@echo "  make dashboard  	- Launch the Streamlit dashboard"
	@echo "  make clean      	- Remove generated files and models"
	@echo "  make clean-data 	- Remove generated data files"
	@echo "  make clean-model 	- Remove generated models"
	@echo "  make help       	- Show this help message"

all:
	make clean
	$(PYTHON) scripts/workflow.py

data:
	$(PYTHON) scripts/data/db.py
	$(PYTHON) scripts/data/prepare_data.py
	$(PYTHON) scripts/data/features.py

train:
	$(PYTHON) scripts/ai/train.py

evaluate:
	$(PYTHON) scripts/ai/evaluate.py

dashboard:
	$(PYTHON) -m streamlit run scripts/dashboard/dashboard.py

clean-data:
	@echo "Cleaning generated data files..."
	rm -rf data/featured/*.parquet 
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Done."

clean-metrics:
	@echo "Cleaning metrics and experiment tracking data..."
	rm -rf data/metrics/*.parquet data/metrics/*.npz data/metrics/*.npy data/mlruns/* data/mlruns/.trash
	@echo "Done."

clean-model:
	@echo "Cleaning generated models..."
	rm -rf data/model/*.txt
	clean-metrics
	@echo "Done."

clean:
	@echo "Cleaning all generated files, models, and tracking data..."
	clean-data
	clean-model
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Done."

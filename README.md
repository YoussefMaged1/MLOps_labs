# MLOps Pipeline Project

<a target="_blank" href="https://cookiecutter-data-science.drivendaily.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Complete MLOps Pipeline for ML Practitioners - An educational project for the ITI MLOps Course.

## Overview

This project demonstrates a complete end-to-end machine learning pipeline using industry-standard tools. It includes data versioning with DVC, configuration management with Hydra, experiment tracking with Weights & Biases (W&B), and model versioning.

## Prerequisites

- Python 3.11+
- Poetry (for dependency management)

Install dependencies:
```bash
poetry install
```

## Project Structure

```
ITI-MLOps/
├── conf/                     # Hydra configuration files
├── data/                     # Data directory
│   ├── external/             # Third-party data
│   ├── interim/              # Intermediate transformed data
│   ├── processed/            # Final processed datasets
│   └── raw/                  # Original immutable data
├── docs/                     # MkDocs documentation
├── models/                   # Trained models and preprocessors
├── notebooks/                # Jupyter notebooks (W&B tutorial)
├── references/               # Data dictionaries and manuals
├── reports/                  # Generated analysis (HTML, PDF, etc.)
├── src/                      # Source code
│   ├── fake/                 # Fake data generators
│   ├── training/             # Training pipeline modules
│   │   ├── process_data.py   # Data preprocessing
│   │   ├── data_training.py # Model training
│   │   └── model_evaluate.py # Model evaluation
│   └── logger.py             # Logging utilities
├── main.py                   # Entry point for training pipeline
├── pyproject.toml            # Project configuration (Poetry)
├── poetry.lock               # Locked dependencies
└── Makefile                  # Convenience commands
```

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| **DVC** | Data version control |
| **Hydra** | Configuration management |
| **W&B** | Experiment tracking & visualization |
| **DagsHub** | ML metadata management |
| **scikit-learn** | Machine learning models |
| **Poetry** | Python dependency management |

## Quick Start

### 1. Data Preparation

Place your raw data in `data/raw/` directory. The pipeline expects a CSV file:
```bash
cp your_data.csv data/raw/Titanic.csv
```

### 2. Run the Pipeline

```bash
python main.py
```

This will:
1. **Process data** - Load raw data, handle missing values, encode features
2. **Train model** - Train a RandomForest classifier
3. **Evaluate** - Output model performance metrics

### 3. View Logs

Logs are stored in `logs/` directory with timestamps.

## Pipeline Details

### Data Processing (`src/training/process_data.py`)

- Loads raw CSV data
- Handles missing values (mean for numerical, mode for categorical)
- Creates preprocessing pipeline (StandardScaler + OneHotEncoder)
- Saves processed data as Parquet files
- Persists preprocessor for inference

### Model Training (`src/training/data_training.py`)

- Reads processed Parquet files
- Extracts features and target
- Trains RandomForest classifier
- Saves model to `models/random_forest_model.pkl`

### Model Evaluation (`src/training/model_evaluate.py`)

- Loads trained model and test data
- Computes accuracy and classification report

## Configuration

Project uses Hydra for configuration. Add config files in `conf/` directory:

```yaml
# conf/config.yaml
model:
  n_estimators: 100
  max_depth: 10
  
data:
  file_name: Titanic
  id_col: PassengerId
  target_col: Survived
```

## Using DVC for Data Versioning

```bash
dvc init
dvc remote add -d myremote s3://your-bucket/path
dvc add data/raw/Titanic.csv
dvc push
```

## Using W&B for Experiment Tracking

The project includes a W&B tutorial notebook:
```bash
jupyter lab notebooks/wandb101.ipynb
```

## Available Make Commands

```bash
make data      # Download/generate data
make train     # Run training
make evaluate  # Evaluate model
make setup     # Setup project
```

## Extending the Project

1. **Add new data sources**: Place in `data/external/`
2. **Add new features**: Modify `src/training/process_data.py`
3. **Add new models**: Add to `src/training/` directory
4. **Add experiments**: Use Hydra configs in `conf/`

## License

MIT License - See LICENSE file.

---
title: Credit Risk API
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Credit Risk Prediction API

A REST API for predicting credit risk using machine learning. Built with FastAPI, MLflow, and deployed on Hugging Face Spaces.

## Overview

This API predicts the probability of loan default for credit applications using a **Gradient Boosting** model trained on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset.
The system processes 48,744 test applications with 56 features, achieving optimal performance with a learned threshold of **0.5064**.

### Key Features

- **Fast inference**: Preprocessed data at startup for sub-second predictions
- **Containerized**: Single-stage Docker image using Micromamba
- **MLflow integration**: Model versioning and metadata management
- **CI/CD pipeline**: Automated testing and deployment via GitHub Actions

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Endpoints  │  │  Services    │  │ Preprocessing│  │
│  │  /predict    │→ │ Prediction   │→ │ Pipeline     │  │
│  │  /data       │  │ Service      │  │ (startup)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│           MLflow Model (Gradient Boosting)              │
│            Threshold: 0.5064 | Size: 477KB              │
├─────────────────────────────────────────────────────────┤
│      Preprocessed Test Data (48,744 applications)       │
│              Memory: ~50MB | Features: 56               │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Runtime** | Python | 3.12 |
| **Web Framework** | FastAPI + Uvicorn | Latest |
| **ML Framework** | scikit-learn | 1.6.1 |
| **Model Management** | MLflow | 2.22.0 |
| **Data Processing** | pandas + numpy | 2.2.2 / 2.0.2 |
| **Container** | Docker + Micromamba | Latest |
| **Deployment** | Hugging Face Spaces | - |

---

## API Documentation

### Endpoints

#### `GET /predict` - Random Client Prediction
Returns a prediction for a randomly selected client from the test dataset.

**Response Example:**
```json
{
  "threshold": 0.5064,
  "probabilityClass0": 0.9156,
  "probabilityClass1": 0.0844,
  "loan_id": 100005,
  "credit_amount": 135000.0,
  "decision": "accepted"
}
```

#### `GET /predict/{loan_id}` - Specific Client Prediction
Returns a prediction for a specific client by loan ID.

**Parameters:**
- `loan_id` (path, integer): Client's loan application ID (100001-456255)

**Response:** Same as `/predict`

**Error Responses:**
- `404`: Loan ID not found
- `500`: Internal server error

#### `GET /data` - Get Preprocessed Data
Returns preprocessed test data for inspection or analysis.

**Query Parameters:**
- `limit` (optional, default=100, max=1000): Number of rows to return
- `offset` (optional, default=0): Starting row index

**Response Example:**
```json
{
  "total_rows": 48744,
  "returned_rows": 100,
  "offset": 0,
  "limit": 100,
  "columns": ["NAME_CONTRACT_TYPE", "CODE_GENDER", ...],
  "data": [
    {
      "SK_ID_CURR": 100001,
      "AMT_CREDIT": 406597.5,
      "AMT_ANNUITY": 24700.5,
      ...
    }
  ]
}
```

#### `GET /data/{loan_id}` - Get Client Features
Returns preprocessed features for a specific client.

**Response Example:**
```json
{
  "loan_id": 100001,
  "features": {
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "male",
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    ...
  }
}
```

#### `GET /healthz` - Health Check
Returns service health status.

**Response:**
```json
{
  "status": "ok",
  "message": "Service is ready."
}
```

#### `GET /ping` - Basic Ping
Simple endpoint for connectivity testing.

**Response:**
```json
{
  "message": "pong"
}
```

### Interactive Documentation

- **Swagger UI**: `/docs` - Interactive API documentation
- **ReDoc**: `/redoc` - Alternative documentation view

---

## Model Information

### Model Details

- **Algorithm**: Gradient Boosting Classifier
- **Framework**: scikit-learn 1.6.1
- **Model Size**: 477 KB (488,534 bytes)
- **Optimal Threshold**: 0.5064 (learned from validation set)
- **Features**: 56 engineered features
- **Training Parameters**:
  - `class_weight`: balanced
  - `l2_regularization`: 0.97
  - `learning_rate`: 0.1315
  - `max_depth`: 7
  - `min_samples_leaf`: 23

### Feature Engineering

The preprocessing pipeline creates the following feature categories:

1. **Financial Ratios** (5 features):
   - `PAYMENT_RATE`: Annuity / Credit
   - `ANNUITY_INCOME_PERC`: Annuity / Income
   - `INCOME_CREDIT_PERC`: Income / Credit
   - `DEBT_TO_INCOME`: Credit / Income
   - `CREDIT_PER_PERSON`: Credit / Family Members

2. **Temporal Features** (5 features):
   - Days since birth (age) - converted to absolute values
   - Days employed
   - Days since registration
   - Days since ID publish
   - Days since last phone change

3. **Categorical Features** (23 features):
   - Contract type, gender, car/realty ownership
   - Income type, education, family status
   - Housing type, occupation, organization type
   - Document flags, contact flags
   - Region/city comparison flags

4. **External Sources** (3 features):
   - `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`

5. **Credit Bureau Inquiries** (6 features):
   - Inquiries per: hour, day, week, month, quarter, year

### Data Transformations

**Critical Preprocessing Steps** (must match training data):

- **Placeholder Replacement**:
  - `DAYS_EMPLOYED`: 365243 → NaN
  - `ORGANIZATION_TYPE`: "XNA" → NaN

- **Missing Values**: 
  - Categorical columns (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`) filled with "Unknown"

- **Gender Standardization** (CRITICAL):
  - `CODE_GENDER`: "XNA" → "male" (matches training preprocessing)
  - `CODE_GENDER`: "M" → "male", "F" → "female"

- **Binary Flags**: Standardized to yes/no format
  - Flag columns (0/1) → "yes"/"no"
  - Car/realty ownership (Y/N) → "yes"/"no"

- **Region Ratings**: 
  - Converted to letter grades (1→A, 2→B, 3→C)
  - Fixed -1 values to 2

- **Temporal Columns**: Negative values converted to absolute

- **Type Casting**: All numeric features cast to `float64` (MLflow requirement)

---

## Deployment

### CI/CD Pipeline (GitHub Actions)

The deployment pipeline runs on every push to `main`:

1. **Quality Checks**:
   - Linting with `flake8` (critical errors + complexity checks)
   - Code formatting validation with `black`

2. **Testing**:
   - Build Docker image
   - Run `pytest` inside container (override ENTRYPOINT)
   - Validate model loading and preprocessing

3. **Deployment**:
   - Push to Hugging Face Spaces using secure credential handling
   - Automatic rebuild and redeployment

### Deployment Configuration

**GitHub Secrets Required:**
- `HF_TOKEN`: Hugging Face API token (write access)
- `HF_USERNAME`: Hugging Face username

**Environment Variables:**
```bash
MLFLOW_TRACKING_URI=file:///tmp/mlruns-disabled
PYTHONPATH=/app:/app/src
MAMBA_ROOT_PREFIX=/opt/conda
PATH=/opt/conda/bin:$PATH
```

### Docker Build Optimization

- **Single-stage build** using `mambaorg/micromamba:latest`
- **Base environment activation**: All packages installed directly to base
- **Git LFS**: Large files (`.pkl`, `.csv`, `.h5`) tracked via Git LFS
- **Line endings**: All text files use LF (configured in `.gitattributes`)

---

## Testing, test Coverage

- Data loading and validation (essential column checks)
- Preprocessing pipeline (transformation logic)
- Model prediction with both ndarray and DataFrame outputs
- Service logic (random/specific client selection)
- Error handling (missing features, invalid IDs)

**Pytest Configuration:**
- Filters Pydantic deprecation warnings from MLflow
- See `pytest.ini` for details

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD pipeline
├── data/
│   └── application_test.csv    # Test dataset (Git LFS, 26MB)
├── models/
│   └── gradient_boosting/
│       ├── MLmodel             # MLflow model metadata
│       ├── model.pkl           # Trained model (Git LFS, 477KB)
│       ├── python_env.yaml     # Model dependencies
│       └── code/               # Model training utilities
│           └── p7_utils/       # (config, logs, metrics)
├── src/
│   └── credit_risk_app/
│       ├── __init__.py
│       ├── main.py             # FastAPI application + lifespan
│       ├── config.py           # Configuration constants
│       ├── preprocessing.py    # Feature engineering (56 features)
│       └── services.py         # PredictionService business logic
├── tests/
│   └── test_minimal.py         # Unit tests
├── Dockerfile                  # Single-stage Micromamba build
├── credit_risk_env.yml         # Conda/Micromamba environment
├── entrypoint.sh              # Container entrypoint (PYTHONPATH setup)
├── pytest.ini                 # Pytest configuration
├── .dockerignore              # Docker build exclusions
├── .gitignore                 # Git tracking exclusions
├── .gitattributes             # Git LFS + line ending config
└── README.md                  # This file
```

---

### Logging

The application uses Python's built-in `logging` module with INFO level:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Informational message")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)
```

**Key Log Messages:**
- Startup: "Full dataset preprocessing completed!"
- Startup metrics: Preprocessing time, shape, memory usage
- Prediction: Loan ID, probabilities, threshold

---

## License

This project is part of an educational assessment and uses data from the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition.

---


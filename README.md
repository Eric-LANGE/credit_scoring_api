---
title: Credit Risk API
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Credit Risk Prediction API

## Overview

This project provides a Credit Risk Prediction tool accessible via a web dashboard and an API. The system uses a machine learning model, managed with MLflow, to predict the probability of a loan applicant defaulting. Users can input client data or request a random client's prediction and view the risk assessment through an interactive dashboard.

## Features

* **Credit Risk Prediction:** Predicts the likelihood of credit default for a given client.
* **Interactive Dashboard:** A web-based interface to visualize prediction scores, client information.
* **REST API:** Provides endpoints for predictions, client data retrieval, and model information.
* **MLflow Integration:** Uses MLflow for model tracking and management.

## Project Structure

```
project_p7/
├── .github/                 # GitHub Actions workflows
├── data/
│   └── application_test.csv # Sample data for testing predictions
├── src/
│   └── credit_risk_app/
│       ├── __init__.py
│       ├── config.py        # Configuration for features, paths, model
│       ├── main.py          # FastAPI application, API endpoints
│       ├── preprocessing.py # Data preprocessing and transformation logic
│       └── services.py      # Business logic for predictions, data loading
├── static/
│   ├── css/
│   │   └── style.css      # CSS for the dashboard
│   └── js/
│       ├── api.js           # JavaScript for API communication (frontend)
│       └── script.js        # JavaScript for dashboard interactivity
├── templates/
│   └── index.html           # HTML structure for the dashboard
├── tests/
│   ├── test_main.py       # Tests for API endpoints
│   └── test_services.py   # Tests for prediction and data services
├── .dockerignore            # Specifies intentionally untracked files for Docker
├── .gitignore               # Specifies intentionally untracked files for Git
├── credit_risk_env.yml      # Conda environment definition
├── Dockerfile               # Docker configuration
├── entrypoint.sh            # Script to run the application
├── pytest.ini               # Pytest configuration
├── README.md                # This file

## Running the Application

There are two primary ways to run this application: locally using Conda/Micromamba for development, or via Docker for production-like deployment.

### 1. Local Development (with Conda/Micromamba)

This method is ideal for development, allowing for features like auto-reloading.

**Prerequisites:**
- Conda or Micromamba is installed.

**Steps:**

1.  **Create and activate the environment:**
    ```bash
    # Using micromamba (recommended)
    micromamba create -f credit_risk_env.yml -y
    micromamba activate credit-risk-env

    # Or using conda
    conda env create -f credit_risk_env.yml
    conda activate credit-risk-env
    ```

2.  **Run the development server:**
    The application source is in the `src/` directory. From the project root, run:
    ```bash
    uvicorn src.credit_risk_app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    - The `--reload` flag automatically restarts the server on code changes.
    - The application will be accessible at [http://localhost:8000](http://localhost:8000).

### 2. Docker Deployment

This is the recommended method for a stable, production-like environment. The `Dockerfile` handles all setup.

**Prerequisites:**
- Docker is installed and running.

**Steps:**

1.  **Build the Docker image:**
    From the project root, run:
    ```bash
    docker build -t credit-risk-app .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 7860:7860 credit-risk-app
    ```
    - This maps port `7860` of your local machine to port `7860` inside the container.
    - The application will be accessible at [http://localhost:7860](http://localhost:7860).

## API Endpoints

The application exposes the following API endpoints through FastAPI:

### Main
* **`GET /`** (HTML): Serves the main interactive dashboard.

### Health Checks
* **`GET /ping`**: A basic health check endpoint. Returns `{"message": "pong"}` if the server is running.
* **`GET /healthz`**: A service-level health check. Returns a `200 OK` status if the prediction service and its resources have loaded successfully, and `503 Service Unavailable` otherwise.

### Predictions
* **`GET /predict`**: Returns a prediction for a randomly selected client from the test dataset.
* **`GET /predict/{loan_id}`**: Returns a prediction for a specific client identified by their `loan_id`.

## Testing

The project uses `pytest` for unit and integration testing. Tests are located in the `tests/` directory:
* `tests/test_main.py`: Contains tests for the FastAPI endpoints.
* `tests/test_services.py`: Contains tests for the business logic in the service layer.

To run the tests, execute the following command from the project root:
```bash
PYTHONPATH=. pytest tests/
```
**Note:** Setting `PYTHONPATH=.` is necessary because the tests import the application code from the `src/` directory (e.g., `from src.credit_risk_app...`). This command adds the project's root directory to Python's path, allowing it to locate the `src` module.



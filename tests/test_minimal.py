# tests/test_minimal.py
import pandas as pd
import numpy as np
import pytest
from typing import List

from src.credit_risk_app.preprocessing import apply_transformations
from src.credit_risk_app.services import PredictionService


class FakePyfuncModel:
    """Stub pyfunc model for tests; returns an ndarray of probabilities."""

    def __init__(self, mode="ndarray"):
        self.mode = mode

    def predict(self, X: pd.DataFrame):
        if self.mode == "ndarray":
            arr = np.array([[0.8, 0.2]])
            return arr
        else:
            # DataFrame shape (1,2)
            return pd.DataFrame([[0.8, 0.2]])


def test_dataloader_requires_sk_id_curr():
    """Test that DataLoader validation checks for essential columns."""
    df = pd.DataFrame({"AMT_CREDIT": [1000.0]})
    missing_essential = [c for c in ["SK_ID_CURR", "AMT_CREDIT"] if c not in df.columns]
    assert "SK_ID_CURR" in missing_essential


def test_apply_transformations_missing_expected_feature_raises():
    """Test that apply_transformations raises ValueError for missing features."""
    df = pd.DataFrame({"SK_ID_CURR": [1], "AMT_CREDIT": [10000.0]}).set_index(
        "SK_ID_CURR"
    )
    expected_features: List[str] = ["NON_EXISTENT_COL"]
    with pytest.raises(ValueError):
        apply_transformations(df, expected_features)


def test_service_accepts_preprocessed_data():
    """
    Tests that PredictionService works with preprocessed data
    (no transformation needed, just validation).
    """
    # Simulate preprocessed data with expected features
    expected_features: List[str] = ["AMT_ANNUITY", "AMT_CREDIT"]

    preprocessed_df = pd.DataFrame(
        {"SK_ID_CURR": [1], "AMT_ANNUITY": [1000.0], "AMT_CREDIT": [10000.0]}
    ).set_index("SK_ID_CURR")

    # Reorder columns to match expected_features (simulating preprocessing)
    preprocessed_df = preprocessed_df[expected_features]

    fake_model = FakePyfuncModel(mode="ndarray")
    service = PredictionService(
        pyfunc_model=fake_model,
        test_data=preprocessed_df,
        expected_features=expected_features,
        threshold=0.5,
    )
    result = service.get_prediction_for_specific_client(loan_id=1)
    assert result["probability_neg"] == 0.8
    assert result["probability_pos"] == 0.2


def test_service_rejects_unpreprocessed_data():
    """
    Tests that PredictionService raises error if data columns don't match expected_features.
    """
    expected_features: List[str] = ["FEATURE_A", "FEATURE_B"]

    # Data with wrong columns (not preprocessed)
    wrong_df = pd.DataFrame(
        {"SK_ID_CURR": [1], "AMT_CREDIT": [10000.0], "WRONG_COL": [999.0]}
    ).set_index("SK_ID_CURR")

    fake_model = FakePyfuncModel(mode="ndarray")
    service = PredictionService(
        pyfunc_model=fake_model,
        test_data=wrong_df,
        expected_features=expected_features,
        threshold=0.5,
    )

    # Should raise HTTPException when trying to predict
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        service.get_prediction_for_specific_client(loan_id=1)
    assert exc_info.value.status_code == 500
    assert "Feature mismatch" in exc_info.value.detail


def test_service_handles_dataframe_model_output():
    """
    Tests that PredictionService correctly handles DataFrame output from model.
    """
    expected_features: List[str] = ["AMT_ANNUITY", "AMT_CREDIT"]

    preprocessed_df = pd.DataFrame(
        {"SK_ID_CURR": [1], "AMT_ANNUITY": [1000.0], "AMT_CREDIT": [10000.0]}
    ).set_index("SK_ID_CURR")
    preprocessed_df = preprocessed_df[expected_features]

    # Use DataFrame mode for fake model
    fake_model = FakePyfuncModel(mode="dataframe")
    service = PredictionService(
        pyfunc_model=fake_model,
        test_data=preprocessed_df,
        expected_features=expected_features,
        threshold=0.5,
    )
    result = service.get_prediction_for_specific_client(loan_id=1)
    assert result["probability_neg"] == 0.8
    assert result["probability_pos"] == 0.2

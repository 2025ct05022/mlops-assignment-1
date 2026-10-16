import os
import pandas as pd
import joblib

def test_cleaned_data_exists():
    """Test if the cleaned dataset was generated and saved correctly."""
    data_path = "artifacts/cleaned_heart_disease.csv"
    assert os.path.exists(data_path), "Cleaned data file does not exist"

    df = pd.read_csv(data_path)
    assert not df.empty, "Cleaned dataset is empty"
    assert 'target' in df.columns, "Target column is missing from dataset"

def test_model_artifacts_exist():
    """Test if the scaler and models were saved successfully."""
    assert os.path.exists("artifacts/models/scaler.pkl"), "Scaler artifact missing"
    assert os.path.exists("artifacts/models/Logistic_Regression.pkl"), "Logistic Regression model missing"
    assert os.path.exists("artifacts/models/Random_Forest.pkl"), "Random Forest model missing"

def test_model_prediction():
    """Test if the saved model can load and make a prediction on a sample input."""
    # Load model and scaler
    model = joblib.load("artifacts/models/Logistic_Regression.pkl")
    scaler = joblib.load("artifacts/models/scaler.pkl")

    # Load a single row of data for testing (excluding target)
    df = pd.read_csv("artifacts/cleaned_heart_disease.csv")
    X_sample = df.drop(columns=['target']).iloc[[0]]

    # Scale and predict
    X_sample_scaled = scaler.transform(X_sample)
    prediction = model.predict(X_sample_scaled)

    assert prediction[0] in [0, 1], "Prediction should be binary (0 or 1)"
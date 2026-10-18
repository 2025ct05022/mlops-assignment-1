from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import time

# Configure logging to write to both the terminal and a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_activity.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Load the saved model and scaler artifacts
model = joblib.load("artifacts/models/Logistic_Regression.pkl")
scaler = joblib.load("artifacts/models/scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        # Accept JSON input
        data = request.get_json()
        logging.info(f"Incoming prediction request: {data}")

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Preprocess features
        X_scaled = scaler.transform(df)

        # Generate prediction and confidence score
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = probabilities[prediction]

        # Calculate processing time
        process_time = round(time.time() - start_time, 4)

        # Log the successful prediction
        logging.info(f"Successful Prediction: {prediction} | Confidence: {confidence:.4f} | Time: {process_time}s")

        # Return prediction and confidence
        return jsonify({
            "prediction": int(prediction),
            "confidence": float(confidence),
            "status": "success"
        })
    except Exception as e:
        logging.error(f"Prediction failed with error: {str(e)}")
        return jsonify({"error": str(e), "status": "failed"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
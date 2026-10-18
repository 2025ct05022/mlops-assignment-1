import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import os
from pathlib import Path

# Create directory to save the final reusable models for Task 4
os.makedirs("artifacts/models", exist_ok=True)

# --- 1. Load Cleaned Data ---
print("Loading cleaned dataset...")
df = pd.read_csv("artifacts/cleaned_heart_disease.csv")
X = df.drop(columns=['target'])
y = df['target']

# --- 2. Feature Engineering ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare final ML features by scaling them
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the preprocessing pipeline to ensure reproducibility
joblib.dump(scaler, "artifacts/models/scaler.pkl")

# --- 3. Model Development & Experiment Tracking ---
# Build the SQLite URI from a filesystem path so spaces in the Windows
# username are not treated as literal "%20" characters by SQLite.
mlflow_db = Path(__file__).resolve().parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
mlflow.set_experiment("Heart_Disease_Prediction")

def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Evaluate using relevant metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Evaluate using crossvalidation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()

        # Log parameters and metrics for experiment tracking
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_mean_accuracy", cv_mean)

        # Save model in a reusable pickle format
        joblib.dump(model, f"artifacts/models/{model_name.replace(' ', '_')}.pkl")

        # MLflow's skops-based serializer flags the internal tree storage used
        # by RandomForest as untrusted. This model was just trained locally, so
        # explicitly trust only that required scikit-learn type when logging it.
        model_log_options = {}
        if isinstance(model, RandomForestClassifier):
            model_log_options["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]

        # Log model in MLflow format
        mlflow.sklearn.log_model(
            model,
            f"{model_name.replace(' ', '_')}_model",
            **model_log_options,
        )

        print(f"--- {model_name} Results ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"CV Mean Accuracy: {cv_mean:.4f}\n")

# Build and train two classification models
lr_model = LogisticRegression(max_iter=1000, random_state=42)
train_and_log_model("Logistic Regression", lr_model, X_train_scaled, y_train, X_test_scaled, y_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_log_model("Random Forest", rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

print("Models saved successfully in artifacts/models/")
print("Run 'python run_mlflow_ui.py' in your terminal to view the experiment tracking dashboard.")
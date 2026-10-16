import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import os

# Create directories for artifacts
os.makedirs("artifacts/plots", exist_ok=True)

# --- 1. Data Acquisition ---
print("Downloading Heart Disease UCI Dataset...")
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

# Combine into a single DataFrame
df = pd.concat([X, y], axis=1)

# --- 2. Data Cleaning & Preprocessing ---
# The UCI dataset target 'num' ranges from 0 to 4.
# Encode feature: Convert to a binary target (0 = no disease, 1 = disease presence)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df = df.drop(columns=['num'])

# Handle missing values (dropping NA for this baseline)
df = df.dropna()
print(f"Dataset shape after cleaning: {df.shape}")

# Save the cleaned dataset
df.to_csv("artifacts/cleaned_heart_disease.csv", index=False)
print("Cleaned dataset saved to artifacts/cleaned_heart_disease.csv")

# --- 3. Exploratory Data Analysis (EDA) ---
sns.set_theme(style="whitegrid")

# A. Class Balance
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target', palette='viridis')
plt.title("Class Balance: Heart Disease Presence (1) vs Absence (0)")
plt.savefig("artifacts/plots/class_balance.png")
plt.close()

# B. Histograms of Features
df.hist(figsize=(15, 12), bins=15, color='teal', edgecolor='black')
plt.suptitle("Feature Histograms", fontsize=16)
plt.tight_layout()
plt.savefig("artifacts/plots/feature_histograms.png")
plt.close()

# C. Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("artifacts/plots/correlation_heatmap.png")
plt.close()

print("EDA visualizations saved in artifacts/plots/")
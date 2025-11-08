#!/usr/bin/env python3
"""
Train posture classification model from collected data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Configuration
DATA_FILE = "data/posture_data.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "posture_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")

# Create models directory
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("POSTURE MODEL TRAINING")
print("="*60)

# Load data
print(f"\nLoading data from {DATA_FILE}...")
try:
    data = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(data)} samples")
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found!")
    print("  Run collect_data.py first to collect training data")
    exit(1)

# Check data distribution
print("\nData distribution:")
label_counts = data['label'].value_counts()
for label, count in label_counts.items():
    print(f"  {label}: {count} samples ({count/len(data)*100:.1f}%)")

# Warn if imbalanced
if len(label_counts) < 2:
    print("\nWarning: Only one class found! Collect data for both 'good' and 'bad' postures")
    exit(1)

min_class = label_counts.min()
max_class = label_counts.max()
if max_class / min_class > 3:
    print(f"\nWarning: Imbalanced dataset (ratio: {max_class/min_class:.1f}:1)")
    print("  Consider collecting more samples for the minority class")

if len(data) < 50:
    print(f"\nWarning: Only {len(data)} samples. Recommended: at least 100 samples")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(0)

# Prepare features and labels
print("\nPreparing features...")
X = data.drop(columns=['label'])
y = data['label']

print(f"Feature dimensions: {X.shape}")
print(f"Features: {X.shape[1]} columns")

# Handle missing values (if any)
if X.isnull().any().any():
    print("Warning: Found missing values, filling with mean")
    X = X.fillna(X.mean())

# Split data
print("\nSplitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Normalize features
print("\nNormalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("Model trained")

# Evaluate on training set
train_score = model.score(X_train_scaled, y_train)
print(f"\nTraining accuracy: {train_score*100:.2f}%")

# Evaluate on test set
test_score = model.score(X_test_scaled, y_test)
print(f"Test accuracy: {test_score*100:.2f}%")

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"Mean CV accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Detailed metrics
print("\n" + "="*60)
print("CLASSIFICATION REPORT (Test Set)")
print("="*60)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n[[TN FP]")
print(" [FN TP]]")

# Feature importance
print("\n" + "="*60)
print("TOP 10 IMPORTANT FEATURES")
print("="*60)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:25s} {row['importance']:.4f}")

# Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
joblib.dump(model, MODEL_FILE)
print(f"Model saved to: {MODEL_FILE}")

joblib.dump(scaler, SCALER_FILE)
print(f"Scaler saved to: {SCALER_FILE}")

# Test loading
print("\nVerifying saved model...")
try:
    loaded_model = joblib.load(MODEL_FILE)
    loaded_scaler = joblib.load(SCALER_FILE)
    test_pred = loaded_model.predict(loaded_scaler.transform(X_test))
    verify_score = (test_pred == y_test).mean()
    print(f"Model loaded successfully, accuracy: {verify_score*100:.2f}%")
except Exception as e:
    print(f"Error loading model: {e}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nModel file: {MODEL_FILE}")
print(f"Scaler file: {SCALER_FILE}")
print(f"Test accuracy: {test_score*100:.2f}%")
print("\nNext steps:")
print("1. Integrate model into main.py")
print("2. Use joblib.load() to load model and scaler")
print("3. Call model.predict(scaler.transform(features))")
print("="*60)

import json
import pickle
import os
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

MODEL_FILE = "/home/theo/smart-posture-assistant/src/nlp_model.pkl"
DATA_FILE = "/home/theo/smart-posture-assistant/src/nlp_data.json"

def preprocess(text):
    return ViTokenizer.tokenize(text.lower())

def train():
    print(f"{GREEN}[INFO] Training SVM model with Vietnamese Tokenizer...{RESET}")
    
    if not os.path.exists(DATA_FILE):
        print(f"{RED}[ERROR] Data file not found: {DATA_FILE}{RESET}")
        return

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X = []
    y = []

    for intent, patterns in data.items():
        for pattern in patterns:
            processed_text = preprocess(pattern)
            X.append(processed_text)
            y.append(intent)

    model = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))
    ])

    model.fit(X, y)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    print(f"{GREEN}[SUCCESS] Smart Model saved to {MODEL_FILE}{RESET}")
    
    test_text = "ngồi có chuẩn không"
    processed_test = preprocess(test_text)
    pred = model.predict([processed_test])[0]
    print(f"[TEST] Input: '{test_text}' -> Tokenized: '{processed_test}' -> Predicted: {pred}")

if __name__ == "__main__":
    train()

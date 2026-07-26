"""
train_model.py
----------------
Trains a spam classifier using TF-IDF features + Multinomial Naive Bayes.

Pipeline:
    raw text -> TF-IDF vectorizer -> Naive Bayes classifier

Why these choices (good to say in an interview):
- TF-IDF turns text into weighted word-importance vectors (down-weights
  common words, up-weights distinctive ones).
- Naive Bayes is a strong, fast baseline for text classification —
  works well even on small datasets and is easy to explain
  (it estimates P(word | spam) vs P(word | ham) using Bayes' theorem).

Run:
    python train_model.py
Produces:
    model/vectorizer.pkl
    model/classifier.pkl
    Prints accuracy, precision, recall, F1 and a confusion matrix.
"""

import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_PATH = "data/spam_dataset.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label_num"]

# 2. Train/test split (stratified so both classes are proportionally represented)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Vectorize text -> TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test_vec)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix (rows=actual, cols=predicted) [ham, spam]:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# 6. Save model + vectorizer for the Flask app to load
with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
    pickle.dump(clf, f)

print(f"\nSaved vectorizer.pkl and classifier.pkl to '{MODEL_DIR}/'")

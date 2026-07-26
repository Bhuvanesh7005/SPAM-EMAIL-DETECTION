"""
app.py
-------
Flask web app that serves the trained spam-detection model.

Routes:
    GET  /              -> UI to type/paste a message and check it
    POST /api/predict    -> REST API: {"message": "..."} -> {"prediction", "confidence"}
    GET  /history         -> page showing past predictions + stats
    POST /api/clear-history -> wipes prediction history

Run:
    python app.py
(Make sure you've run train_model.py first so model/*.pkl exist.)
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os

import database

app = Flask(__name__)

MODEL_DIR = "model"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")

# ---- Load trained model artifacts at startup ----
if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(CLASSIFIER_PATH)):
    raise FileNotFoundError(
        "Model files not found. Run `python train_model.py` first "
        "(after `python generate_dataset.py` if data/spam_dataset.csv is missing)."
    )

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(CLASSIFIER_PATH, "rb") as f:
    classifier = pickle.load(f)

database.init_db()


def predict_message(text):
    vec = vectorizer.transform([text])
    pred = classifier.predict(vec)[0]              # 0 = ham, 1 = spam
    proba = classifier.predict_proba(vec)[0]        # [P(ham), P(spam)]
    label = "spam" if pred == 1 else "ham"
    confidence = float(proba[1] if pred == 1 else proba[0])
    return label, round(confidence * 100, 2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Message text is required"}), 400

    label, confidence = predict_message(message)
    database.save_prediction(message, label, confidence)

    return jsonify({
        "message": message,
        "prediction": label,
        "confidence": confidence
    })


@app.route("/history")
def history():
    records = database.get_history()
    stats = database.get_stats()
    return render_template("history.html", records=records, stats=stats)


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    database.clear_history()
    return jsonify({"message": "History cleared"})


if __name__ == "__main__":
    app.run(debug=True)

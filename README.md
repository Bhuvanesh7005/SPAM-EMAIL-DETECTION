# Spam Message Detector (ML + Full-Stack)

A complete project: an ML model (TF-IDF + Naive Bayes) trained to detect
spam messages, served through a Flask web app with a clean UI and a
SQLite-backed prediction history. Ideal for TCS NQT / placement interviews
— small enough to explain fully, but covers the whole ML + web pipeline.

## What it does
- Type or paste any SMS/email text → get an instant **Spam / Not Spam**
  prediction with a confidence score
- Every prediction is saved to a database
- A **History** page shows past checks + overall stats (total, spam, ham)

## Project structure
```
spam-detector-fullstack/
├── generate_dataset.py   # builds data/spam_dataset.csv (synthetic, realistic spam/ham messages)
├── train_model.py        # trains TF-IDF + Naive Bayes, saves model/*.pkl, prints metrics
├── database.py           # sqlite3 helper (init, save, fetch history/stats)
├── app.py                # Flask app: loads model, serves UI + REST API
├── data/
│   └── spam_dataset.csv
├── model/
│   ├── vectorizer.pkl     # trained TF-IDF vectorizer
│   └── classifier.pkl     # trained Naive Bayes model
├── templates/             # index.html, history.html, base.html
├── static/                # css/js
├── requirements.txt
└── predictions.db         # created automatically on first run
```

## How to run
```bash
cd spam-detector-fullstack
pip install -r requirements.txt

# (Optional — the dataset and trained model are already included)
python generate_dataset.py   # regenerate the training data
python train_model.py        # retrain the model, prints accuracy/precision/recall/F1

python app.py                 # start the web app
```
Open **http://127.0.0.1:5000**.

## How the ML pipeline works
1. **Data**: `generate_dataset.py` creates ~300 labeled messages using
   common spam patterns (prize/lottery, urgency, phishing links, fake
   loan offers) and everyday ham messages, with randomized details so the
   model can't just memorize exact strings.
   *(In a real-world version you'd swap this for the public "SMS Spam
   Collection Dataset" from UCI/Kaggle — mention this if asked.)*
2. **Feature extraction**: `TfidfVectorizer` converts each message into a
   weighted word/bigram vector — common words (the, is, you) get
   down-weighted, distinctive spam words (free, winner, urgent, click)
   get up-weighted.
3. **Model**: `MultinomialNB` (Naive Bayes) — a fast, interpretable
   baseline that estimates P(word | spam) vs P(word | ham) using Bayes'
   theorem, then combines them for a full message.
4. **Evaluation**: train/test split (80/20, stratified) with accuracy,
   precision, recall, F1, and a confusion matrix printed to console.
5. **Serving**: Flask loads the saved `vectorizer.pkl` + `classifier.pkl`
   once at startup and reuses them for every request (fast — no
   retraining per request).

## REST API
```
POST /api/predict
Body: { "message": "You won a free prize! Click here" }
Response: { "message": "...", "prediction": "spam", "confidence": 91.4 }
```

## Talking points for interviews
- **Why TF-IDF + Naive Bayes instead of a deep learning model?** For
  small/medium text datasets, Naive Bayes is fast to train, doesn't
  overfit easily, and is easy to explain in an interview — a great
  "practical baseline" answer. You can mention that in production you'd
  compare it against Logistic Regression / SVM / a fine-tuned
  transformer, and pick based on accuracy vs latency needs.
- **Precision vs Recall trade-off**: for spam detection, false positives
  (a real message marked as spam) are usually worse than false negatives
  — you'd rather tune the model to be conservative about flagging spam.
- **Full pipeline ownership**: you can talk through data → features →
  model → evaluation → API → UI → persistence, which is exactly what
  interviewers want to hear you can do end-to-end.
- **Persistence layer**: uses raw `sqlite3` (not an ORM) — shows you can
  write SQL directly, not just rely on abstractions.
- **Model serialization**: `pickle` to save/load the trained vectorizer +
  classifier so the app doesn't retrain on every request — a basic but
  important MLOps concept.

## Natural extensions (good answers to "what would you improve?")
- Swap in the real UCI SMS Spam Collection dataset for more realistic accuracy numbers
- Add user accounts so history is per-user
- Try Logistic Regression / SVM / a fine-tuned DistilBERT and compare metrics
- Add a `/api/metrics` endpoint exposing live accuracy on a labeled test set
- Containerize with Docker and deploy (Render/Railway/AWS)

# CODSOFT – Task 4: Spam SMS Detection

This project builds a TF‑IDF + classic ML pipeline to classify SMS as **spam** or **ham**.

## Quick Start

```bash
# 1) Install dependencies (example)
pip install -U scikit-learn pandas joblib matplotlib

# 2) Train (will auto‑choose best of Naive Bayes, Logistic Regression, Linear SVM):
python train_spam_sms.py --input spam.csv --output spam_sms_pipeline.joblib
```

## Use the saved pipeline

```python
import joblib
pipe = joblib.load("spam_sms_pipeline.joblib")
pipe.predict(["Win a free iPhone by clicking this link!"])
# -> array(['spam'], dtype='<U4')
```

## Notes
- Text pre‑processing replaces URLs, emails, and numbers and lowercases text.
- Vectorization: `TfidfVectorizer` with English stopwords and bigrams.
- Models tried: Naive Bayes, Logistic Regression, Linear SVM (with probability calibration).
- The best model (by F1 on the **spam** class) is saved automatically.

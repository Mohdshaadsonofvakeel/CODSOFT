#!/usr/bin/env python3
# CODSOFT – Task: Spam SMS Detection
# Train TF-IDF + NaiveBayes pipeline and save it to disk.
import re, os, sys, joblib, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

url_pat = re.compile(r"(https?://\S+|www\.\S+)")
email_pat = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
num_pat = re.compile(r"\b\d+\b")
non_alnum = re.compile(r"[^a-z0-9\s]")

def text_clean(x: str) -> str:
    if not isinstance(x, str):
        x = "" if x is None else str(x)
    x = x.lower()
    x = url_pat.sub(" URL ", x)
    x = email_pat.sub(" EMAIL ", x)
    x = num_pat.sub(" NUM ", x)
    x = non_alnum.sub(" ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def build_pipelines():
    pipes = {}
    pipes["NaiveBayes"] = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=text_clean, stop_words="english", ngram_range=(1,2), min_df=2)),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    pipes["LogisticRegression"] = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=text_clean, stop_words="english", ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear"))
    ])
    pipes["LinearSVM"] = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=text_clean, stop_words="english", ngram_range=(1,2), min_df=2)),
        ("clf", CalibratedClassifierCV(LinearSVC(), cv=5))
    ])
    return pipes

def main(args):
    df = pd.read_csv(args.input, encoding="latin-1")
    if set(["v1","v2"]).issubset(df.columns):
        df = df[["v1","v2"]].rename(columns={"v1":"label","v2":"text"})
    else:
        df = df.iloc[:, :2]
        df.columns = ["label","text"]
    df = df.dropna(subset=["label","text"]).drop_duplicates(subset=["label","text"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham","spam"])].copy()

    X_train, X_test, y_train, y_test = train_test_split(df["text"].values, df["label"].values, test_size=0.2, random_state=42, stratify=df["label"].values)

    pipes = build_pipelines()
    rows = []
    best_name, best_pipe, best_f1 = None, None, -1.0
    for name, pipe in pipes.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, pos_label="spam", average="binary", zero_division=0)
        rows.append((name, acc, p, r, f1))
        if f1 > best_f1:
            best_name, best_pipe, best_f1 = name, pipe, f1

    print("Model comparison (acc, prec_spam, rec_spam, f1_spam):")
    for name, acc, p, r, f1 in sorted(rows, key=lambda r: r[-1], reverse=True):
        print(f" - {name}: acc={acc:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")

    # Save best
    out = args.output or f"spam_sms_{best_name.lower()}_tfidf_pipeline.joblib"
    joblib.dump(best_pipe, out)
    print("Saved:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="spam.csv", help="Path to spam.csv")
    parser.add_argument("--output", default=None, help="Output .joblib path")
    args = parser.parse_args()
    main(args)

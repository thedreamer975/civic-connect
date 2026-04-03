"""
worker_insights.py
Worker Performance Insights pipeline:
- Generates synthetic feedback
- Trains sentiment classifier (sentence-transformers embeddings + LogisticRegression)
- Fits BERTopic for topic modeling
- Produces worker scorecards & improvement suggestions
- Saves pipeline and artifacts (joblib + .csv)
"""

import random
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------- 1) Synthetic dataset generator ----------
WORKER_IDS = [f"worker_{i+1}" for i in range(10)]   # adjust number of workers
N_SAMPLES = 2000

POS_TEMPLATES = [
    "The worker fixed the issue quickly and was polite.",
    "Great job: problem solved same day, very professional.",
    "Thanks — timely resolution and polite staff.",
    "Worker was efficient and explained the fix clearly."
]
NEG_TEMPLATES = [
    "The worker arrived late and the issue remains unresolved.",
    "Poor workmanship; problem came back the next day.",
    "Rude behavior and slow service, not satisfied.",
    "Took too long and didn't clean the area afterwards."
]
NEUTRAL_TEMPLATES = [
    "Work completed, nothing special to report.",
    "Issue resolved, standard service.",
    "Worker did what was expected, no extra comments.",
    "Fixed the problem; average experience."
]
TOPIC_OVERLAYS = {
    "safety": ["dangerous", "hazardous", "risk", "injury", "unsafe"],
    "timing": ["late", "delayed", "hours", "time", "wait"],
    "quality": ["poor", "came back", "not fixed", "patch", "reopened"],
    "cleanliness": ["mess", "trash", "not cleaned", "left debris"],
    "communication": ["rude", "no update", "didn't explain", "no response"]
}

def synthesize_feedback(n=N_SAMPLES):
    rows = []
    for i in range(n):
        worker = random.choice(WORKER_IDS)
        p = random.random()
        if p < 0.55:
            text = random.choice(POS_TEMPLATES)
            label = "positive"
        elif p < 0.85:
            text = random.choice(NEUTRAL_TEMPLATES)
            label = "neutral"
        else:
            text = random.choice(NEG_TEMPLATES)
            label = "negative"
        if random.random() < 0.4:
            topic = random.choice(list(TOPIC_OVERLAYS.keys()))
            word = random.choice(TOPIC_OVERLAYS[topic])
            text = text + f" It was {word} regarding {topic}."
        rating = {
            "positive": random.randint(4,5),
            "neutral": random.randint(3,4),
            "negative": random.randint(1,3)
        }[label]
        timestamp = (datetime.utcnow() - timedelta(days=random.randint(0,90))).isoformat()
        rows.append({
            "worker_id": worker,
            "feedback_text": text,
            "sentiment_label": label,
            "rating": rating,
            "timestamp": timestamp
        })
    return pd.DataFrame(rows)

def train_sentiment(df, embed_model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(embed_model_name)
    texts = df["feedback_text"].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    y = df["sentiment_label"].map({"negative": 0, "neutral": 1, "positive": 2}).values
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("=== Sentiment classification report ===")
    print(classification_report(y_test, y_pred, target_names=["negative","neutral","positive"]))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/embedding_model.joblib")
    joblib.dump(clf, "models/sentiment_clf.joblib")
    return model, clf

def train_topic_model(df, embed_model):
    docs = df["feedback_text"].tolist()
    topic_model = BERTopic(embedding_model=embed_model, verbose=True, nr_topics="auto")
    topics, probs = topic_model.fit_transform(docs)
    os.makedirs("models", exist_ok=True)
    topic_model.save("models/bertopic_model")
    return topic_model, topics, probs

SUGGESTION_TEMPLATES = {
    "safety": "Ensure safety protocols: use cones/signage and PPE, escalate high-risk cases immediately.",
    "timing": "Improve scheduling and on-time arrival: track ETA, notify citizens proactively.",
    "quality": "Provide better fix quality: additional training, use durable materials, follow-up checks.",
    "cleanliness": "Ensure site cleanup after repair; carry trash bags and tools for cleanup.",
    "communication": "Improve communication: call before arrival, explain the fix, and provide status updates.",
    "default": "Follow up with citizen, verify fix quality, and document action taken."
}

def build_scorecards(df, topic_model, topics):
    df2 = df.copy().reset_index(drop=True)
    df2["topic"] = topics
    topic_info = topic_model.get_topic_info()
    topic_map = {}
    for _, row in topic_info.iterrows():
        t_id = int(row["Topic"])
        if t_id == -1:
            continue
        words = topic_model.get_topic(t_id)
        top_words = " ".join([w for w, _ in words])
        label = "default"
        for k in TOPIC_OVERLAYS.keys():
            if any(s in top_words for s in TOPIC_OVERLAYS[k]):
                label = k
                break
        topic_map[t_id] = {"words": top_words, "label": label}
    scorecards = []
    for worker, group in df2.groupby("worker_id"):
        avg_rating = group["rating"].mean()
        sentiment_counts = group["sentiment_label"].value_counts(normalize=True).to_dict()
        t_counts = group["topic"].value_counts().to_dict()
        labeled_topics = {}
        for t_id, cnt in t_counts.items():
            if int(t_id) in topic_map:
                lbl = topic_map[int(t_id)]["label"]
            else:
                lbl = "default"
            labeled_topics[lbl] = labeled_topics.get(lbl, 0) + cnt
        if labeled_topics:
            top_issue = max(labeled_topics.items(), key=lambda x: x[1])[0]
        else:
            top_issue = "default"
        suggestion = SUGGESTION_TEMPLATES.get(top_issue, SUGGESTION_TEMPLATES["default"])
        scorecards.append({
            "worker_id": worker,
            "avg_rating": round(avg_rating, 2),
            "sentiment_distribution": sentiment_counts,
            "top_issue_area": top_issue,
            "suggestion": suggestion
        })
    return scorecards, topic_map

def main():
    print("Generating synthetic feedback dataset...")
    df = synthesize_feedback(N_SAMPLES)
    df.to_csv("synthetic_worker_feedback.csv", index=False)
    print("Saved synthetic_worker_feedback.csv")
    print("Training sentiment classifier (embeddings + logistic regression)...")
    embed_model, clf = train_sentiment(df)
    print("Training BERTopic for topic modeling (this may take a bit)...")
    topic_model, topics, probs = train_topic_model(df, embed_model)
    print("Building worker scorecards...")
    scorecards, topic_map = build_scorecards(df, topic_model, topics)
    with open("worker_scorecards.json", "w") as f:
        json.dump(scorecards, f, indent=2)
    print("Saved worker_scorecards.json")
    for s in scorecards[:5]:
        print(s)
    with open("topic_map.json", "w") as f:
        json.dump(topic_map, f, indent=2)
    df["topic"] = topics
    df.to_csv("synthetic_worker_feedback_with_topics.csv", index=False)
    print("Saved synthetic_worker_feedback_with_topics.csv")

if __name__ == "__main__":
    main()

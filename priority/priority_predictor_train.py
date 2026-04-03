"""
priority_predictor_train.py
Synthetic data + Priority Prediction baseline
Generates synthetic civic complaint records and trains a TF-IDF + RandomForest pipeline
that predicts priority: high / medium / low.
"""
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- 1) Synthetic dataset generator ----------
ISSUE_TYPES = ["pothole", "drainage", "streetlight", "garbage"]
LOCATION_TYPES = ["highway", "main_road", "residential", "alley"]

TEMPLATES = {
    "pothole": [
        "Large pothole on the road, car damage likely",
        "Deep pothole causing vibration to vehicles",
        "Small pothole near the curb",
        "Pothole filled with water, hidden hazard",
        "Multiple potholes across the lane"
    ],
    "drainage": [
        "Drain clogged, water overflowing onto road",
        "Open drainage cover, dangerous for pedestrians",
        "Standing water near drain after rain",
        "Drainage blocked with garbage and leaves",
        "Sewage backup overflowing onto street"
    ],
    "streetlight": [
        "Streetlight not working at night, area dark",
        "Flickering streetlight, sparks visible",
        "Pole lamp broken and leaning",
        "Light out on the main road intersection",
        "Streetlight intermittent, safety concern"
    ],
    "garbage": [
        "Overflowing garbage bin, foul smell",
        "Garbage dumped near sidewalk, attracting animals",
        "Scattered trash on street after collection missed",
        "Pile of household waste blocking pedestrian path",
        "Illegal dumping near drainage"
    ]
}

SEVERITY_PHRASES = [
    "urgent", "immediate attention needed", "hazardous", "causing accidents",
    "please fix soon", "low priority", "not urgent", "needs quick fix"
]

def synthesize_text(issue_type):
    base = random.choice(TEMPLATES[issue_type])
    extra = []
    if random.random() < 0.4:
        extra.append(random.choice(SEVERITY_PHRASES))
    if random.random() < 0.15:
        extra.append(random.choice(SEVERITY_PHRASES))
    if extra:
        return base + ". " + " ".join(extra)
    return base

def label_priority(row):
    issue = row["issue_type"]
    loc = row["location_type"]
    dur = row["duration_days"]
    desc = row["description"].lower()
    if any(k in desc for k in ["sparks", "fire", "injury", "hazardous", "sewage", "overflowing", "blocking", "accident"]):
        return "high"
    if issue == "drainage" and ("overflow" in desc or "sewage" in desc or dur > 3):
        return "high"
    if issue == "pothole":
        if loc in ["highway", "main_road"] and dur > 7:
            return "high"
        if dur > 30:
            return "medium"
        if dur < 3:
            return "low"
    if issue == "streetlight":
        if "flicker" in desc or "sparks" in desc or loc == "highway":
            return "high"
        if dur > 14:
            return "medium"
    if issue == "garbage":
        if ("overflow" in desc or loc == "main_road") and dur > 2:
            return "high"
        if dur < 2:
            return "low"
    if dur <= 2:
        return "low"
    if dur <= 10:
        return "medium"
    return "high"

def generate_synthetic_dataset(n_samples=3000):
    rows = []
    for _ in range(n_samples):
        issue = random.choice(ISSUE_TYPES)
        loc = random.choice(LOCATION_TYPES)
        duration = int(np.random.exponential(scale=7.0))
        duration = max(0, min(duration, 365))
        desc = synthesize_text(issue)
        rows.append({
            "issue_type": issue,
            "location_type": loc,
            "duration_days": duration,
            "description": desc
        })
    df = pd.DataFrame(rows)
    df["priority"] = df.apply(label_priority, axis=1)
    return df

def train_priority_model(df):
    X = df[["description", "issue_type", "location_type", "duration_days"]]
    y = df["priority"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    text_transformer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(transformers=[
        ("tfidf", text_transformer, "description"),
        ("ohe", ohe, ["issue_type", "location_type"]),
        ("scale", StandardScaler(), ["duration_days"])
    ], remainder="drop")
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1)
    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", clf)
    ])
    print("Training model...")
    pipe.fit(X_train, y_train)
    print("Training complete.")
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring="f1_weighted", n_jobs=-1)
    print(f"CV weighted-F1 scores: {cv_scores}, mean={cv_scores.mean():.3f}")
    y_pred = pipe.predict(X_test)
    print("\nClassification report on test set:")
    print(classification_report(y_test, y_pred, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=["high", "medium", "low"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["high", "medium", "low"], yticklabels=["high", "medium", "low"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig("priority_confusion_matrix.png")
    print("Saved confusion matrix to priority_confusion_matrix.png")
    joblib.dump(pipe, "priority_pipeline.joblib")
    print("Saved trained pipeline to priority_pipeline.joblib")
    return pipe

def demo_predict(pipe, n=6):
    samples = [
        {"issue_type": "pothole", "location_type": "highway", "duration_days": 10, "description": "Large pothole on lane causing car damage. urgent"},
        {"issue_type": "garbage", "location_type": "residential", "duration_days": 1, "description": "Small pile of trash near curb. not urgent"},
        {"issue_type": "drainage", "location_type": "main_road", "duration_days": 2, "description": "Drain clogged and overflowing onto road, immediate attention needed"},
        {"issue_type": "streetlight", "location_type": "residential", "duration_days": 20, "description": "Streetlight not working at night"},
        {"issue_type": "streetlight", "location_type": "main_road", "duration_days": 1, "description": "Flickering light, sparks visible"},
        {"issue_type": "pothole", "location_type": "alley", "duration_days": 0, "description": "Shallow pothole near sidewalk"}
    ]
    df = pd.DataFrame(samples)
    preds = pipe.predict(df)
    probs = pipe.predict_proba(df)
    for i, row in df.iterrows():
        print(f"\nSample #{i}: {row.to_dict()}")
        print(f"Predicted priority: {preds[i]}  (probabilities: {dict(zip(pipe.classes_, probs[i]))})")

if __name__ == "__main__":
    df = generate_synthetic_dataset(n_samples=3000)
    print("Sample of generated data:")
    print(df.sample(5).to_dict(orient="records")[:3])
    df.to_csv("synthetic_priority_dataset.csv", index=False)
    print("Saved synthetic_priority_dataset.csv")
    pipe = train_priority_model(df)
    demo_predict(pipe)

# Build a pipeline for civic complaint management:
# 1. Load complaints dataset (complaint_id, complaint_text, metadata).
# 2. Use sentence-transformers (all-MiniLM-L6-v2) to embed complaints.
# 3. Use FAISS for fast similarity search to detect duplicates (similarity > 0.75).
# 4. Group duplicates together and count frequency of each cluster.
# 5. Merge duplicate count + original features into the Priority Prediction Model.
# 6. Train priority model and predict complaint urgency scores.
# 7. Output each complaint with priority score and duplicate cluster id.

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------
# Step 1: Load Complaints
# -----------------------------------------
data = pd.DataFrame({
    "complaint_id": [1, 2, 3, 4, 5],
    "complaint_text": [
        "Huge pothole on main road near school",
        "Streetlight not working at park",
        "Large pothole in front of the school",
        "Garbage not collected from market area",
        "Broken streetlight near central park"
    ],
    "location": ["school road", "park", "school road", "market", "park"],
    "category": ["pothole", "streetlight", "pothole", "garbage", "streetlight"],
    "timestamp": pd.to_datetime([
        "2025-09-01 08:00", "2025-09-01 09:30",
        "2025-09-01 08:15", "2025-09-02 10:00",
        "2025-09-01 09:45"
    ])
})

# -----------------------------------------
# Step 2: Embedding Model (semantic upgrade)
# Options: "all-mpnet-base-v2" or "multi-qa-mpnet-base-dot-v1"
# -----------------------------------------
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(data["complaint_text"].tolist(), convert_to_numpy=True)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# -----------------------------------------
# Step 3: Hybrid Features (text + metadata)
# -----------------------------------------
# Convert timestamps to numeric (seconds since min timestamp)
min_time = data["timestamp"].min()
data["time_numeric"] = (data["timestamp"] - min_time).dt.total_seconds()

# Encode location similarity (dummy: exact match = 1, else 0)
location_sim = np.zeros((len(data), len(data)))
for i in range(len(data)):
    for j in range(len(data)):
        location_sim[i, j] = 1 if data.loc[i, "location"] == data.loc[j, "location"] else 0

# Category similarity (dummy: same category = 1, else 0)
category_sim = np.zeros((len(data), len(data)))
for i in range(len(data)):
    for j in range(len(data)):
        category_sim[i, j] = 1 if data.loc[i, "category"] == data.loc[j, "category"] else 0

# Cosine similarity from embeddings
text_sim = cosine_similarity(embeddings)

# Fuse all similarities (weighted sum)
combined_sim = (0.6 * text_sim) + (0.2 * location_sim) + (0.2 * category_sim)

# -----------------------------------------
# Step 4: Clustering for Duplicates
# -----------------------------------------
# Agglomerative clustering with similarity threshold
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.4,   # controls sensitivity (lower = stricter)
    metric="precomputed",
    linkage="average"
)

clusters = clustering.fit_predict(1 - combined_sim)  # distance = 1 - similarity
data["cluster_id"] = clusters

# Count duplicates per cluster
dup_count = pd.Series(data["cluster_id"]).value_counts().to_dict()
data["duplicate_count"] = data["cluster_id"].map(lambda x: dup_count.get(x, 1))

# -----------------------------------------
# Step 5: Priority Prediction Model
# -----------------------------------------
# Example priority labels
data["priority_label"] = [2, 1, 2, 0, 1]

X = pd.get_dummies(data[["category", "duplicate_count"]], drop_first=True)
y = data["priority_label"]

priority_model = RandomForestClassifier(n_estimators=100, random_state=42)
priority_model.fit(X, y)

data["predicted_priority"] = priority_model.predict(X)

# -----------------------------------------
# Step 6: Output Results
# -----------------------------------------
print("\nEnhanced Complaint Table:")
print(data[["complaint_id", "complaint_text", "cluster_id", "duplicate_count", "predicted_priority"]])

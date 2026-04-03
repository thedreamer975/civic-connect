import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

DATA_FILE = "data/complaints.json"
if not os.path.exists("data"):
    os.makedirs("data")

if os.path.exists(DATA_FILE):
    complaints = json.load(open(DATA_FILE))
else:
    complaints = []

dimension = 384
index = faiss.IndexFlatL2(dimension)

# Rebuild index with existing complaints
if complaints:
    embeddings = [np.array(c["embedding"]) for c in complaints]
    index.add(np.vstack(embeddings).astype("float32"))

def check_duplicate(complaint):
    """Check if complaint is duplicate using FAISS."""
    emb = model.encode([complaint["raw_text"]])[0].astype("float32")
    if index.ntotal > 0:
        D, I = index.search(np.array([emb]), 1)
        if D[0][0] < 0.6:  # threshold
            return True, complaints[I[0][0]]["cluster_id"]
    # new cluster
    cluster_id = len(complaints) + 1
    return False, cluster_id

def save_complaint(complaint, cluster_id):
    """Save new complaint + embedding."""
    emb = model.encode([complaint["raw_text"]])[0].astype("float32")
    complaint["embedding"] = emb.tolist()
    complaint["cluster_id"] = cluster_id
    complaints.append(complaint)
    index.add(np.array([emb]))
    with open(DATA_FILE, "w") as f:
        json.dump(complaints, f, indent=2)

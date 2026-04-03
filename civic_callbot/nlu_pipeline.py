from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def process_complaint(text):
    """Extract complaint type, priority, and location heuristically."""
    complaint = {"raw_text": text, "type": None, "priority": "normal", "location": None}
    # Detect type
    if re.search(r"pothole|गड्ढा|bache", text, re.I):
        complaint["type"] = "pothole"
    elif re.search(r"garbage|कचरा|basura", text, re.I):
        complaint["type"] = "garbage"
    elif re.search(r"light|लाइट|farola", text, re.I):
        complaint["type"] = "streetlight"
    elif re.search(r"drain|नाली|drenaje", text, re.I):
        complaint["type"] = "drainage"
    # Detect urgency
    if re.search(r"urgent|immediate|तुरंत|जरूरी", text, re.I):
        complaint["priority"] = "urgent"
    return complaint

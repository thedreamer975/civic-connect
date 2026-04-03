from flask import Flask, jsonify
import json

app = Flask(__name__)

@app.route("/complaints")
def get_complaints():
    with open("data/complaints.json") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    app.run(port=6060, debug=True)

"""
Flask app to handle incoming Twilio calls for civic complaints.
"""

from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse
import os
from stt import transcribe_audio
from nlu_pipeline import process_complaint
from duplicate_detector import check_duplicate, save_complaint

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Civic Callbot is running. Use /voice for Twilio webhook."

@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()
    resp.say("Welcome to Civic Complaint Helpline.", language="en-IN")
    resp.say("नमस्ते! अपनी भाषा चुनें: हिंदी के लिए 1 दबाएँ, English के लिए 2 दबाएँ", language="hi-IN")
    resp.gather(input="speech dtmf", timeout=5, num_digits=1, action="/handle-language")
    return str(resp)

@app.route("/handle-language", methods=["POST"])
def handle_language():
    choice = request.values.get("Digits")
    resp = VoiceResponse()
    lang_map = {"1": "hi-IN", "2": "en-IN", "3": "te-IN", "4": "ta-IN"}
    lang = lang_map.get(choice, "en-IN")
    resp.say("Please describe your complaint after the beep.", language=lang)
    resp.record(maxLength=30, action="/process-complaint", transcribe=False)
    return str(resp)

@app.route("/process-complaint", methods=["POST"])
def process_complaint():
    recording_url = request.values.get("RecordingUrl")
    text = transcribe_audio(recording_url)
    complaint = process_complaint(text)
    duplicate, cluster_id = check_duplicate(complaint)
    if duplicate:
        msg = f"Your complaint matches an existing issue. Cluster ID {cluster_id}."
    else:
        save_complaint(complaint, cluster_id)
        msg = "Your complaint has been logged successfully."
    resp = VoiceResponse()
    resp.say(msg, language="en-IN")
    return str(resp)

if __name__ == "__main__":
    app.run(port=5000, debug=True)

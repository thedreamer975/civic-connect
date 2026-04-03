import openai
import requests
import tempfile
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(url):
    """Download Twilio audio and transcribe with Whisper."""
    audio = requests.get(url + ".wav").content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio)
        f.flush()
        with open(f.name, "rb") as audio_file:
            transcript = openai.Audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
    return transcript

# test_client.py: Test FastAPI pothole detection endpoint
import requests
import sys

API_URL = 'http://localhost:8000/infer/'
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else 'sample.jpg'

with open(IMAGE_PATH, 'rb') as f:
    files = {'file': (IMAGE_PATH, f, 'image/jpeg')}
    response = requests.post(API_URL, files=files)
    print('Response:', response.json())

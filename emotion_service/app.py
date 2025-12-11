import os
import tempfile
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    f = request.files["audio"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save temp file
    suffix = os.path.splitext(f.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        # Read file bytes for API
        with open(tmp_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
             return jsonify({"error": "HF_TOKEN not set"}), 500

        # API URL
        api_url = "https://api-inference.huggingface.co/models/prithivMLmods/Speech-Emotion-Classification"
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        response = requests.post(api_url, headers=headers, data=audio_bytes)
        
        if response.status_code != 200:
             return jsonify({"error": f"HF API Error: {response.text}"}), response.status_code

        # The API returns a list of dicts: [{'label': 'TAG', 'score': 0.99}, ...]
        predictions = response.json()
        
        # Flatten if it's a list(list(dict)) which sometimes happens with HF inference API
        if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], list):
            predictions = predictions[0]

        # Sort just in case
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
        
        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

import os
import logging
import pandas as pd
import requests
import traceback
import csv
from flask import Flask, request, jsonify, send_file, send_from_directory
from io import BytesIO
import google.generativeai as genai
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from config import Config


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)


# "origins": ",os.getenv("CORS_ALLOWED_ORIGINS "").split(",") if os.getenv("CORS_ALLOWED_ORIGINS") else [],
#            "origins": os.getenv("CORS_ALLOWED_ORIGINS").split(","),

load_dotenv()
# Initialize directories
Config.init_directories()

# Configure CORS
CORS(
    app,
    resources={
        r"/*": {
            "origins": os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(","),
            "methods": os.getenv("CORS_ALLOWED_METHODS", "GET,POST,OPTIONS,PUT,DELETE").split(","),
            "allow_headers": os.getenv("CORS_ALLOWED_HEADERS", "Content-Type,Authorization").split(","),
            "supports_credentials": os.getenv("CORS_SUPPORTS_CREDENTIALS", "true").lower() == "true",
        }
    }
)
#CORS(app,resources={
#        r"/*": {
#            "origins": ["http://localhost:3000"],
#            "methods": ["GET", "POST", "OPTIONS"],
#            "allow_headers": ["Content-Type", "Authorization"],
#        }
#    },
#    supports_credentials=True,
#)

# Constants
#BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
#SPECIES_DATA_DIR = os.path.join(BASE_DATA_DIR, "MMAforays")
#UPLOADS_DIR = os.path.join(BASE_DATA_DIR, "uploads")
#PRONUNCIATION_CACHE_FILE = os.path.join(BASE_DATA_DIR, "pronounce.csv")
#INITIAL_FILE_PATH = os.path.join(UPLOADS_DIR, "macleod-obs-taxa.csv")

# Global state for current file
current_file: Dict[str, Any] = {
    "path": None,
    "directory": None,
    "data": None,
}

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Helper Functions
def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load CSV data and return a shuffled DataFrame."""
    try:
        data = pd.read_csv(file_path)[["image_url", "scientific_name", "common_name"]].dropna()
        return data.sample(frac=1).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return None

def save_pronunciation_cache(cache: Dict[str, str]) -> None:
    """Save pronunciation cache to a CSV file."""
    try:
        with open(Config.PRONUNCIATION_CACHE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scientific_name", "pronunciation"])
            for name, pronunciation in cache.items():
                writer.writerow([name, pronunciation])
    except Exception as e:
        logger.error(f"Error saving pronunciation cache: {str(e)}")

def load_pronunciation_cache() -> Dict[str, str]:
    """Load pronunciation cache from a CSV file."""
    cache = {}
    if os.path.exists(Config.PRONUNCIATION_CACHE_FILE):
        try:
            with open(Config.PRONUNCIATION_CACHE_FILE, "r") as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) == 2:
                        cache[row[0]] = row[1]
        except Exception as e:
            logger.error(f"Error loading pronunciation cache: {str(e)}")
    return cache

# Load initial data
if os.path.exists(Config.INITIAL_FILE_PATH):
    current_file["path"] = Config.INITIAL_FILE_PATH
    current_file["directory"] = "uploads"
    current_file["data"] = load_csv_data(Config.INITIAL_FILE_PATH)
else:
    logger.warning(f"Initial file not found: {Config.INITIAL_FILE_PATH}")
    current_file["data"] = pd.DataFrame(columns=["image_url", "scientific_name", "common_name"])

# Load pronunciation cache
pronunciation_cache = load_pronunciation_cache()

# Flask Routes
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path: str):
    """Serve the React app."""
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/data")
@cross_origin()
def get_data():
    """Example API endpoint."""
    return {"message": "Hello from Flask!"}

@app.route("/get_hints", methods=["GET"])
def get_hints():
    """Get hints for the current flashcard."""
    try:
        if current_file["data"] is None:
            default_file = os.path.join(Config.UPLOADS_DIR, "myspecies.csv")
            current_file["data"] = load_csv_data(default_file)
            current_file["path"] = default_file
            current_file["directory"] = "uploads"

        if current_file["data"] is not None:
            hints = current_file["data"]["scientific_name"].unique().tolist()
            return jsonify(hints), 200
        return jsonify({"error": "No data available"}), 404
    except Exception as e:
        logger.error(f"Error in get_hints: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/check_answer", methods=["POST"])
def check_answer():
    """Check if the user's answer is correct."""
    payload = request.json
    user_answer = payload.get("answer", "").strip().lower()
    card = payload.get("card", {})

    if "scientific_name" not in card:
        return jsonify({"correct": False, "message": "Invalid card data received."}), 400

    if user_answer == card["scientific_name"].lower():
        common_name = card.get("common_name", "")
        return jsonify({"correct": True, "message": f"Correct! ({common_name})"}), 200

    return jsonify({"correct": False, "message": "Incorrect. Try again!"}), 200

# Add new route to load all cards at once
@app.route("/load_cards", methods=["POST"])
def load_cards():
    """Load all cards from a CSV file."""
    payload = request.json
    filename = payload.get("filename")
    directory = payload.get("directory", "MMAforays")

    try:
        file_path = os.path.join(Config.BASE_DATA_DIR, directory, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        data = load_csv_data(file_path)
        if data is None:
            return jsonify({"error": "Failed to load CSV file"}), 500

        # Convert DataFrame to list of dictionaries
        cards = data.to_dict('records')
        return jsonify(cards), 200

    except Exception as e:
        logger.error(f"Error in load_cards: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Modify the load_csv_data function to return all data without shuffling
def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load CSV data and return DataFrame."""
    try:
        data = pd.read_csv(file_path)[["image_url", "scientific_name", "common_name"]].dropna()
        return data
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return None

@app.route("/current_file_info", methods=["GET"])
def get_current_file_info():
    """Get information about the current file."""
    if current_file["path"]:
        return jsonify(
            {
                "filename": os.path.basename(current_file["path"]),
                "directory": current_file["directory"],
            }
        ), 200
    return jsonify({"error": "No file currently selected"}), 404

@app.route("/get_image", methods=["GET"])
def get_image():
    """Fetch an image from a URL."""
    url = request.args.get("url")
    logger.info(f"Attempting to fetch image from URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        logger.info(f"Image fetch successful. Content-Type: {response.headers.get('Content-Type')}")
        return send_file(BytesIO(response.content), mimetype=response.headers.get("Content-Type", "image/jpeg"))
    except Exception as e:
        logger.error(f"Error fetching image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/list_csv_files", methods=["GET"])
def list_csv_files():
    """List CSV files in a directory."""
    directory = request.args.get("directory", "MMAforays")
    if directory == "MMAforays":
        directory_path = Config.SPECIES_DATA_DIR
    else:
        directory_path = Config.UPLOADS_DIR
#    directory_path = os.path.join(Config.BASE_DATA_DIR, directory)
    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    return jsonify({"files": csv_files}), 200

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Upload a CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    directory = request.form.get("directory", "uploads")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".csv"):
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.BASE_DATA_DIR, directory, filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully"}), 200

    return jsonify({"error": "Invalid file type"}), 400

@app.route("/select_csv", methods=["POST"])
def select_csv():
    """Select a CSV file for flashcards."""
    global current_file
    payload = request.json
    filename = payload.get("filename")
    directory = payload.get("directory", "MMAforays")

    try:
        file_path = os.path.join(Config.BASE_DATA_DIR, directory, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        new_data = load_csv_data(file_path)
        if new_data is None:
            return jsonify({"error": "Failed to load CSV file"}), 500

        current_file["path"] = file_path
        current_file["directory"] = directory
        current_file["data"] = new_data

        return jsonify(
            {
                "message": "CSV file selected successfully",
                "first_card": new_data.iloc[0].to_dict(),
            }
        ), 200
    except Exception as e:
        logger.error(f"Error in select_csv: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/delete_csv/<filename>", methods=["DELETE"])
def delete_csv(filename: str):
    """Delete a CSV file."""
    if not filename.endswith(".csv"):
        return jsonify({"error": "Invalid file type"}), 400

    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        os.remove(file_path)
        return jsonify({"message": "File deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pronounce_name", methods=["POST"])
def pronounce_name():
    """Get pronunciation for a scientific name."""
    global pronunciation_cache
    payload = request.json
    scientific_name = payload.get("scientific_name", "")

    if scientific_name in pronunciation_cache:
        return jsonify({"pronunciation": pronunciation_cache[scientific_name]}), 200

    try:
        prompt = f"Pronounce {scientific_name} using English Scientific Latin with explanation"
        response = model.generate_content(prompt)
        pronunciation = response.text

        pronunciation_cache[scientific_name] = pronunciation
        save_pronunciation_cache(pronunciation_cache)

        return jsonify({"pronunciation": pronunciation}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/exit_application", methods=["POST"])
def exit_application():
    """Exit the application."""
    try:
        if pronunciation_cache:
            save_pronunciation_cache(pronunciation_cache)
        os._exit(0)
    except Exception as e:
        logger.error(f"Exit application error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
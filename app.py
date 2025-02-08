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
import fcntl
import csv
from typing import Dict

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
            "origins": os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
            #            "methods": os.getenv("CORS_ALLOWED_METHODS", "GET,POST,OPTIONS,PUT,DELETE").split(","),
            #            "allow_headers": os.getenv("CORS_ALLOWED_HEADERS", "Content-Type,Authorization").split(","),
            #            "supports_credentials": os.getenv("CORS_SUPPORTS_CREDENTIALS", "true").lower() == "true",
        }
    }
)

# Global state for current file
current_file: Dict[str, Any] = {
    "path": None,
    "directory": None,
    "data": None,
}


# Helper Functions
def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load CSV data and return a shuffled DataFrame."""
    try:
        data = pd.read_csv(file_path)[["image_url", "scientific_name", "common_name"]].dropna()
        return data.sample(frac=1).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return None


# Load initial data
if os.path.exists(Config.INITIAL_FILE_PATH):
    current_file["path"] = Config.INITIAL_FILE_PATH
    current_file["directory"] = "uploads"
    current_file["data"] = load_csv_data(Config.INITIAL_FILE_PATH)
else:
    logger.warning(f"Initial file not found: {Config.INITIAL_FILE_PATH}")
    current_file["data"] = pd.DataFrame(columns=["image_url", "scientific_name", "common_name"])


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


# Load pronunciation cache
pronunciation_cache = load_pronunciation_cache()

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")


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


# Flask Routes
@app.route('/wakeup', methods=['GET'])
def wakeup():
    return "Server is awake!", 200

# Add this endpoint to your Flask app
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

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
    directory = payload.get("directory", "mmaforays")

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
        # ,"observer_name",
        #                                        "observation_year","observation_url"
        data = pd.read_csv(file_path)[
            ["image_url", "scientific_name", "common_name", "taxa_url", "observer_name", "observation_year",
             "observation_url"]].dropna()
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
    """List CSV files in a directory in alphabetical order."""
    directory = request.args.get("directory", "mmaforays")
    if directory == "mmaforays":
        directory_path = Config.SPECIES_DATA_DIR
    else:
        directory_path = Config.UPLOADS_DIR

    # Get CSV files and sort them alphabetically
    csv_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".csv")], key=str.lower)
 #   csv_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".csv")])
    return jsonify({"files": csv_files}), 200


def get_taxon_id(scientific_name):
    """Fetch the taxon_id for a given scientific name from iNaturalist."""
    url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": scientific_name, "rank": "species"}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            return results[0].get("id")

    print(f"No taxon_id found for {scientific_name}.")
    return None


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Upload a CSV file, add a taxa_url column, and save only specific columns."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    directory = request.form.get("directory", "uploads")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".csv"):
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.BASE_DATA_DIR, directory, filename)

        # Read the uploaded CSV file directly from the request
        rows = []
        file_stream = file.stream.read().decode("utf-8").splitlines()
        reader = csv.DictReader(file_stream)
        for row in reader:
            # Fetch the taxon_id and construct the taxa_url
            scientific_name = row.get("scientific_name", "")
            taxon_id = get_taxon_id(scientific_name)
            taxa_url = f"https://www.inaturalist.org/taxa/{taxon_id}" if taxon_id else "N/A"

            # Create a new row with only the required columns
            new_row = {
                "scientific_name": scientific_name,
                "common_name": row.get("common_name", "N/A"),
                "image_url": row.get("image_url", "N/A"),
                "taxa_url": taxa_url
            }
            rows.append(new_row)

        # Save the modified data to a new CSV file
        output_filename = f"{filename}"
        output_filepath = os.path.join(Config.BASE_DATA_DIR, directory, output_filename)
        with open(output_filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["scientific_name", "common_name", "image_url", "taxa_url"])
            writer.writeheader()
            writer.writerows(rows)

        return jsonify({"message": "File uploaded and modified successfully", "filename": output_filename}), 200

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/select_csv", methods=["POST"])
def select_csv():
    """Select a CSV file for flashcards."""
    global current_file
    payload = request.json
    filename = payload.get("filename")
    directory = payload.get("directory", "mmaforays")

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


import fcntl
import csv
import os
from typing import Dict


def save_single_pronunciation(scientific_name: str, pronunciation: str) -> bool:
    """
    Append a single pronunciation record to the cache file with file locking.
    Returns True if successful, False otherwise.
    """
    try:
        # Open file in append mode with file locking
        with open(Config.PRONUNCIATION_CACHE_FILE, 'a', newline='') as f:
            # Acquire an exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.writer(f)
                # Only write the new record
                writer.writerow([scientific_name, pronunciation])
                return True
            finally:
                # Release the lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Error saving pronunciation to cache: {str(e)}")
        return False


@app.route("/pronounce_name", methods=["POST"])
def pronounce_name():
    """Get pronunciation for a scientific name."""
    global pronunciation_cache
    payload = request.json
    scientific_name = payload.get("scientific_name", "")

    if not scientific_name:
        return jsonify({"error": "Scientific name is required"}), 400

    # Check cache first
    if scientific_name in pronunciation_cache:
        return jsonify({"pronunciation": pronunciation_cache[scientific_name]}), 200

    try:
        # Generate pronunciation using Gemini
        prompt = f"Pronounce {scientific_name} using English Scientific Latin with explanation"
        response = model.generate_content(prompt)
        pronunciation = response.text

        # Update both cache and file
        pronunciation_cache[scientific_name] = pronunciation

        # Try to save to file
        if save_single_pronunciation(scientific_name, pronunciation):
            return jsonify({"pronunciation": pronunciation}), 200
        else:
            # If save failed, still return the pronunciation but log the error
            logger.warning("Failed to save pronunciation to cache file")
            return jsonify({
                "pronunciation": pronunciation,
                "warning": "Pronunciation generated but not cached"
            }), 200

    except Exception as e:
        logger.error(f"Error in pronounce_name: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Optional: Function to initialize the cache file if it doesn't exist
def initialize_pronunciation_cache_file():
    """Create the pronunciation cache file with headers if it doesn't exist."""
    if not os.path.exists(Config.PRONUNCIATION_CACHE_FILE):
        try:
            with open(Config.PRONUNCIATION_CACHE_FILE, 'w', newline='') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.writer(f)
                    writer.writerow(["scientific_name", "pronunciation"])
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Error initializing pronunciation cache file: {str(e)}")


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

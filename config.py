# config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Environment
    ENV = os.getenv('FLASK_ENV', 'development')
#    ENV = os.environ.get('FLASK_ENV', 'development')
    # Base paths
    BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    # Subdirectories
    SPECIES_DATA_DIR = os.path.join(BASE_DATA_DIR, "mmaforays")
    UPLOADS_DIR = os.path.join(BASE_DATA_DIR, "uploads")

    PRONUNCIATION_CACHE_FILE = os.path.join(BASE_DATA_DIR, "pronounce.csv")
    INITIAL_FILE_PATH = os.path.join(UPLOADS_DIR, "macleod-obs-taxa.csv")

    @classmethod
    def init_directories(cls):
        """Ensure all required directories exist."""
        os.makedirs(cls.SPECIES_DATA_DIR, exist_ok=True)
        os.makedirs(cls.UPLOADS_DIR, exist_ok=True)

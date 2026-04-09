"""
Configuration for Gemma 4 Vision Studio.
Handles API keys, model settings, and image constraints.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")

# Model Settings
DEFAULT_MODEL = "gemma-4-31b-it"
VISION_MODEL = "gemma-4-31b-it"  # Supports multimodal input

# Image Constraints
MAX_IMAGE_SIZE_MB = 10  # Maximum image size in MB
MAX_IMAGE_DIMENSION = 4096  # Maximum dimension (width/height)
ALLOWED_FORMATS = ["jpg", "jpeg", "png", "gif", "webp"]
MIN_IMAGE_DIMENSION = 32  # Minimum dimension

# API Settings
API_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Segmentation Settings
SEGMENTATION_MODEL = "facebook/detr-resnet-50"  # DETR model for segmentation
SEGMENTATION_THRESHOLD = 0.5  # Confidence threshold

# HTML Generation Settings
HTML_MAX_LENGTH = 5000  # Maximum HTML code length
HTML_EXTRACTION_PATTERN = r"<html.*?>.*?</html>"  # Pattern for HTML extraction

# App Settings
APP_HOST = "0.0.0.0"
APP_PORT = 8000
CORS_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000"]

# Mock mode for testing without API key
MOCK_MODE = not GEMINI_API_KEY  # Enable mock mode if no API key

def get_api_key() -> str:
    """Get the Gemini API key."""
    return GEMINI_API_KEY

def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return MOCK_MODE

def validate_image_size(size_bytes: int) -> bool:
    """Validate image size is within limits."""
    return size_bytes <= MAX_IMAGE_SIZE_MB * 1024 * 1024

def validate_image_format(filename: str) -> bool:
    """Validate image format is allowed."""
    ext = filename.lower().split(".")[-1]
    return ext in ALLOWED_FORMATS
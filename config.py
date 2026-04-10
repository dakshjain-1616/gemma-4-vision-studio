"""
Configuration for Gemma 4 Vision Studio.
Handles API keys, model settings, and image constraints.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── API Provider Configuration ───────────────────────────────────────────────
# Set OPENROUTER_API_KEY to use OpenRouter (recommended — supports Gemma 3/4 with vision)
# Set GEMINI_API_KEY to use Google AI Studio directly
# If neither is set, the app runs in mock mode (no real AI, demo responses only)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Google AI Studio key

# Auto-select provider — OpenRouter takes priority if both keys are present
if OPENROUTER_API_KEY:
    API_PROVIDER = "openrouter"
elif GEMINI_API_KEY:
    API_PROVIDER = "google"
else:
    API_PROVIDER = None  # triggers mock mode

# ── OpenRouter Settings ──────────────────────────────────────────────────────
# OpenRouter provides Gemma models (and 100+ others) via an OpenAI-compatible API.
# Get a key at https://openrouter.ai
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")

# ── Google AI Settings ───────────────────────────────────────────────────────
# Direct access to Google's generative language API.
# Get a key at https://aistudio.google.com/app/apikey
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")

# Legacy aliases (used in tests + other modules)
DEFAULT_MODEL = GEMINI_MODEL
VISION_MODEL = GEMINI_MODEL

# ── Image Constraints ────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 32
ALLOWED_FORMATS = ["jpg", "jpeg", "png", "gif", "webp"]

# ── API Settings ─────────────────────────────────────────────────────────────
API_TIMEOUT = 30        # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1         # seconds between retries

# ── Segmentation Settings ────────────────────────────────────────────────────
SEGMENTATION_MODEL = "facebook/detr-resnet-50"
SEGMENTATION_THRESHOLD = 0.5

# ── HTML Generation Settings ─────────────────────────────────────────────────
HTML_MAX_LENGTH = 5000
HTML_EXTRACTION_PATTERN = r"<html.*?>.*?</html>"

# ── App Settings ─────────────────────────────────────────────────────────────
APP_HOST = "0.0.0.0"
APP_PORT = 8000
CORS_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000"]

# ── Mock Mode ────────────────────────────────────────────────────────────────
# Automatically enabled when no API key is set
MOCK_MODE = API_PROVIDER is None


def get_api_key() -> str:
    """Return the active API key."""
    if API_PROVIDER == "openrouter":
        return OPENROUTER_API_KEY
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

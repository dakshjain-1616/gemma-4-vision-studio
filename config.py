"""
Configuration for Gemma 4 Vision Studio.
Handles API keys, model settings, and image constraints.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Inference Backend Configuration ──────────────────────────────────────────
# Set INFERENCE_BACKEND to explicitly choose a provider:
#   'ollama'     → Local Ollama server (http://localhost:11434)
#   'llamacpp'   → Local llama.cpp server (http://localhost:8080)
#   'openrouter' → OpenRouter API (cloud)
#   'google'     → Google AI Studio API (cloud)
#   'mock'       → Demo mode (no AI, fake responses)
# If not set, provider is auto-selected from API keys (OpenRouter → Google → mock)
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "").lower()

# ── API Provider Configuration ───────────────────────────────────────────────
# Set OPENROUTER_API_KEY to use OpenRouter (recommended — supports Gemma 3/4 with vision)
# Set GEMINI_API_KEY to use Google AI Studio directly
# If neither is set, the app runs in mock mode (no real AI, demo responses only)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Google AI Studio key

# Auto-select provider — INFERENCE_BACKEND takes priority, then OpenRouter, then Google, then mock
if INFERENCE_BACKEND in ("ollama", "llamacpp", "openrouter", "google", "mock"):
    API_PROVIDER = INFERENCE_BACKEND
elif OPENROUTER_API_KEY:
    API_PROVIDER = "openrouter"
elif GEMINI_API_KEY:
    API_PROVIDER = "google"
else:
    API_PROVIDER = "mock"  # triggers mock mode

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

# ── Ollama Settings ───────────────────────────────────────────────────────────
# Local inference via Ollama (https://ollama.com)
# Run: ollama pull gemma4  (tags: gemma4:e2b, gemma4:e4b, gemma4:27b)
# Then: ollama serve
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")

# ── llama.cpp Settings ────────────────────────────────────────────────────────
# Local inference via llama.cpp server (https://github.com/ggerganov/llama.cpp)
# Download GGUF from: https://huggingface.co/unsloth/gemma-4-27B-it-GGUF
# Run: ./llama-server -m gemma-4-27b-it-Q4_K_M.gguf --port 8080
LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080")
LLAMACPP_MODEL = os.getenv("LLAMACPP_MODEL", "gemma-4-27b-it-Q4_K_M.gguf")

# Legacy aliases (used in tests + other modules)
DEFAULT_MODEL = GEMINI_MODEL
VISION_MODEL = GEMINI_MODEL

# ── Image Constraints ────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 32
ALLOWED_FORMATS = ["jpg", "jpeg", "png", "gif", "webp"]

# ── API Settings ─────────────────────────────────────────────────────────────
API_TIMEOUT = 120       # seconds (CPU inference needs more time)
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
# Automatically enabled when no API key is set and no local backend configured
MOCK_MODE = API_PROVIDER == "mock"


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

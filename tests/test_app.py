"""
Pytest tests for Gemma 4 Vision Studio API endpoints.
Uses mocks to test without actual API calls.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import app after path setup
from app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_gemma_client():
    """Mock GemmaClient for testing."""
    with patch('app.gemma_client') as mock:
        mock.analyze_image.return_value = {
            "success": True,
            "text": "Mock analysis result",
            "mock_mode": True
        }
        mock.generate_html_from_screenshot.return_value = {
            "success": True,
            "html_code": "<html><body>Mock HTML</body></html>",
            "mock_mode": True
        }
        mock.analyze_for_segments.return_value = {
            "success": True,
            "objects": [{"description": "mock object"}],
            "mock_mode": True
        }
        yield mock


@pytest.fixture
def mock_segmenter():
    """Mock ImageSegmenter for testing."""
    with patch('app.segmenter') as mock:
        mock.draw_annotations.return_value = {
            "success": True,
            "detections": [
                {"label": "person", "confidence": 0.95, "box": [10, 10, 100, 100]}
            ],
            "image_base64": "mock_base64_image",
            "model_used": "Mock"
        }
        yield mock


@pytest.fixture
def mock_screenshot_to_html():
    """Mock ScreenshotToHTML for testing."""
    with patch('app.screenshot_to_html') as mock:
        mock.convert.return_value = {
            "success": True,
            "html_code": "<html><head><title>Test</title></head><body><h1>Mock</h1></body></html>",
            "is_valid": True,
            "mock_mode": True
        }
        yield mock


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "mock_mode" in data
    assert "version" in data


def test_root_endpoint(client):
    """Test root endpoint serves HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_analyze_image_success(client, mock_gemma_client):
    """Test image analysis endpoint with valid image."""
    # Create a simple test image
    import base64
    from PIL import Image
    from io import BytesIO
    
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {"file": ("test.png", img_bytes, "image/png")}
    
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "text" in data
    assert data["mock_mode"] is True


def test_analyze_image_with_prompt(client, mock_gemma_client):
    """Test image analysis with custom prompt."""
    import base64
    from PIL import Image
    from io import BytesIO
    
    img = Image.new('RGB', (100, 100), color='blue')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {"file": ("test.png", img_bytes, "image/png")}
    data = {"prompt": "Describe in detail"}
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 200


def test_analyze_image_no_file(client):
    """Test image analysis without file returns error."""
    response = client.post("/analyze")
    assert response.status_code in (400, 422)


def test_analyze_image_invalid_format(client):
    """Test image analysis with invalid format."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400


def test_segment_image_success(client, mock_segmenter):
    """Test segmentation endpoint with valid image."""
    from PIL import Image
    from io import BytesIO
    
    img = Image.new('RGB', (100, 100), color='green')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {"file": ("test.png", img_bytes, "image/png")}
    
    response = client.post("/segment", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "detections" in data
    assert len(data["detections"]) > 0


def test_segment_image_no_file(client):
    """Test segmentation without file returns error."""
    response = client.post("/segment")
    assert response.status_code in (400, 422)


def test_screenshot_to_html_success(client, mock_screenshot_to_html):
    """Test screenshot to HTML endpoint."""
    from PIL import Image
    from io import BytesIO
    
    img = Image.new('RGB', (100, 100), color='white')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {"file": ("screenshot.png", img_bytes, "image/png")}
    
    response = client.post("/screenshot-to-html", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "html_code" in data
    assert data["is_valid"] is True


def test_screenshot_to_html_no_file(client):
    """Test screenshot to HTML without file returns error."""
    response = client.post("/screenshot-to-html")
    assert response.status_code in (400, 422)


@pytest.fixture
def mock_pipeline_processor():
    """Mock PipelineProcessor for testing."""
    with patch('app.pipeline_processor') as mock:
        mock.analyze_structured.return_value = {
            "success": True,
            "description": "A mock dashboard with analytics widgets.",
            "detected_elements": [
                {"element_type": "header", "description": "Top navigation", "text_content": "Dashboard", "confidence": 0.95},
                {"element_type": "chart", "description": "Line chart", "text_content": None, "confidence": 0.88},
            ],
            "categories": ["dashboard", "analytics"],
            "text_content": "Dashboard Analytics",
            "question": None,
            "answer": None,
            "mock_mode": True,
        }
        yield mock


def test_pipeline_success(client, mock_pipeline_processor):
    """Test pipeline endpoint returns structured analysis."""
    from PIL import Image
    from io import BytesIO

    img = Image.new('RGB', (100, 100), color='blue')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/pipeline", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "description" in data
    assert "detected_elements" in data
    assert "categories" in data
    assert len(data["detected_elements"]) == 2


def test_pipeline_with_question(client, mock_pipeline_processor):
    """Test pipeline endpoint with optional Q&A question."""
    from PIL import Image
    from io import BytesIO

    mock_pipeline_processor.analyze_structured.return_value["question"] = "What is shown?"
    mock_pipeline_processor.analyze_structured.return_value["answer"] = "A dashboard."

    img = Image.new('RGB', (100, 100), color='green')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    files = {"file": ("test.png", img_bytes, "image/png")}
    data_form = {"question": "What is shown?"}
    response = client.post("/pipeline", files=files, data=data_form)
    assert response.status_code == 200
    d = response.json()
    assert d["success"] is True
    assert d["answer"] == "A dashboard."


def test_pipeline_no_file(client):
    """Test pipeline without file returns error."""
    response = client.post("/pipeline")
    assert response.status_code in (400, 422)


def test_pipeline_processor_class():
    """Test PipelineProcessor instantiation."""
    from pipeline_processor import PipelineProcessor
    processor = PipelineProcessor()
    assert processor.client is not None

    # Test JSON parsing fallback
    result = processor._parse_structured('not json at all')
    assert "description" in result
    assert isinstance(result["detected_elements"], list)

    # Test valid JSON parsing
    import json
    valid = json.dumps({"description": "test", "detected_elements": [], "categories": ["x"], "text_content": ""})
    parsed = processor._parse_structured(valid)
    assert parsed["description"] == "test"


def test_error_handling(client):
    """Test general error handling."""
    # Test with invalid endpoint
    response = client.get("/nonexistent")
    assert response.status_code == 404


# Test config module
def test_config_imports():
    """Test that config module imports correctly."""
    from config import (
        GEMINI_MODEL,
        MAX_IMAGE_SIZE_MB,
        MOCK_MODE,
        ALLOWED_FORMATS
    )
    assert GEMINI_MODEL == "gemma-4-31b-it"
    assert MAX_IMAGE_SIZE_MB == 10
    assert isinstance(MOCK_MODE, bool)
    assert isinstance(ALLOWED_FORMATS, list)


def test_gemma_client_mock_mode():
    """Test GemmaClient instantiation in mock mode."""
    from gemma_client import GemmaClient
    
    client = GemmaClient()
    assert client.mock_mode is True  # No API key set
    
    # Test mock response
    result = client._mock_response({"contents": [{"parts": [{"text": "test"}]}]})
    assert "candidates" in result


def test_screenshot_to_html_class():
    """Test ScreenshotToHTML class instantiation."""
    from screenshot_to_html import ScreenshotToHTML
    
    converter = ScreenshotToHTML()
    assert converter.client is not None
    
    # Test HTML validation
    valid_html = "<html><head></head><body></body></html>"
    assert converter._validate_html(valid_html)

    invalid_html = "not html"
    assert not converter._validate_html(invalid_html)


def test_segmenter_class():
    """Test ImageSegmenter class instantiation."""
    from segmenter import ImageSegmenter
    
    segmenter = ImageSegmenter()
    assert segmenter.threshold == 0.5
    # Should use fallback since DETR won't be installed in test env
    assert segmenter.use_gemma_fallback is True

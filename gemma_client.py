"""
GemmaClient - Client for Google Gemini API with Gemma 4 models.
Provides image analysis, HTML generation from screenshots, and segmentation analysis.
Includes mock fallback for testing without API key.
"""

import base64
import json
import os
import time
from typing import Dict, List, Optional, Any

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_BASE_URL,
    DEFAULT_MODEL,
    API_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    MOCK_MODE,
    HTML_MAX_LENGTH,
    HTML_EXTRACTION_PATTERN,
)


class GemmaClient:
    """Client for interacting with Google Gemini API using Gemma 4 models."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the GemmaClient.

        Args:
            api_key: Gemini API key (optional, uses env var if not provided)
            model: Model to use (optional, uses default if not provided)
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self.mock_mode = MOCK_MODE or not self.api_key
        self.base_url = GEMINI_BASE_URL

    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _get_mime_type(self, image_path: str) -> str:
        """
        Get the MIME type of an image based on its extension.

        Args:
            image_path: Path to the image file

        Returns:
            MIME type string
        """
        ext = image_path.lower().split(".")[-1]
        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        return mime_types.get(ext, "image/jpeg")

    def _make_api_request(self, payload: Dict) -> Optional[Dict]:
        """
        Make a request to the Gemini API.

        Args:
            payload: Request payload

        Returns:
            API response as dictionary or None if failed
        """
        if self.mock_mode:
            return self._mock_response(payload)

        try:
            import urllib.request

            url = f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}

            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
            )

            for attempt in range(MAX_RETRIES):
                try:
                    with urllib.request.urlopen(req, timeout=API_TIMEOUT) as response:
                        return json.loads(response.read().decode("utf-8"))
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"API request failed: {e}")
            return None

    def _mock_response(self, payload: Dict) -> Dict:
        """
        Generate a mock response for testing without API key.

        Args:
            payload: Request payload

        Returns:
            Mock response dictionary
        """
        # Check if this is an HTML generation request
        prompt = payload.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
        prompt_lower = prompt.lower()
        
        prompt = prompt_lower
        # Structured pipeline analysis
        if "element_type" in prompt or ("json" in prompt and "detected_elements" in prompt):
            return {
                "candidates": [{
                    "content": {"parts": [{"text": json.dumps({
                        "description": "A modern analytics dashboard with a clean, professional design featuring data visualization components and navigation elements.",
                        "detected_elements": [
                            {"element_type": "navigation", "description": "Top navigation bar with logo and menu links", "text_content": "Dashboard · Reports · Settings · Profile", "confidence": 0.95},
                            {"element_type": "header", "description": "Page title with subtitle and date range picker", "text_content": "Analytics Overview — Last 30 Days", "confidence": 0.93},
                            {"element_type": "card", "description": "KPI metric card showing total users", "text_content": "12,543 Users ↑ 8.2%", "confidence": 0.94},
                            {"element_type": "card", "description": "KPI metric card showing revenue", "text_content": "$48,290 Revenue ↑ 12.1%", "confidence": 0.91},
                            {"element_type": "chart", "description": "Line chart visualising daily active users over time", "text_content": None, "confidence": 0.89},
                            {"element_type": "table", "description": "Data table with sortable columns listing recent events", "text_content": "Event | User | Date | Status", "confidence": 0.87},
                            {"element_type": "button", "description": "Primary CTA export button in the top right", "text_content": "Export CSV", "confidence": 0.92},
                            {"element_type": "input", "description": "Search field for filtering table rows", "text_content": "Search events...", "confidence": 0.88}
                        ],
                        "categories": ["dashboard", "analytics", "web-ui", "data-visualization", "admin-panel"],
                        "text_content": "Dashboard · Reports · Settings · Profile · Analytics Overview · Last 30 Days · 12,543 Users · $48,290 Revenue · Export CSV · Search events..."
                    })}]}
                }]
            }
        if "html" in prompt or "code" in prompt or "website" in prompt:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": """<html>
<head>
    <title>Mock Website</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #333; color: white; padding: 10px; }
        .content { padding: 20px; }
    </style>
</head>
<body>
    <div class="header">Mock Header</div>
    <div class="content">
        <h1>Welcome</h1>
        <p>This is a mock response generated without API key.</p>
    </div>
</body>
</html>"""
                                }
                            ]
                        }
                    }
                ]
            }
        elif "segment" in prompt or "object" in prompt or "detect" in prompt:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "Detected objects: person, car, building. Segmentation available for these objects."
                }
                            ]
                        }
                    }
                ]
            }
        else:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "This is a mock response. The image appears to contain various elements. In mock mode, detailed analysis is not available."
                                }
                            ]
                        }
                    }
                ]
            }

    def analyze_image(self, image_path: str, prompt: str = "Describe this image") -> Dict:
        """
        Analyze an image using the Gemini API.

        Args:
            image_path: Path to the image file
            prompt: Prompt for image analysis

        Returns:
            Dictionary containing analysis results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image_base64 = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": image_base64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024,
            }
        }

        response = self._make_api_request(payload)
        
        if response and "candidates" in response:
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            return {"success": True, "text": text, "mock_mode": self.mock_mode}
        
        return {"success": False, "text": "Failed to analyze image", "mock_mode": self.mock_mode}

    def generate_html_from_screenshot(self, image_path: str) -> Dict:
        """
        Generate HTML code from a screenshot image.

        Args:
            image_path: Path to the screenshot image

        Returns:
            Dictionary containing generated HTML code
        """
        prompt = """Convert this screenshot into clean, semantic HTML code. 
        Include proper structure with head, body, and appropriate semantic tags.
        Add inline CSS styles to match the visual design.
        Output only the HTML code, no explanations."""

        result = self.analyze_image(image_path, prompt)
        
        if result["success"]:
            # Extract HTML code from response
            import re
            html_match = re.search(HTML_EXTRACTION_PATTERN, result["text"], re.DOTALL | re.IGNORECASE)
            if html_match:
                html_code = html_match.group(0)
            else:
                html_code = result["text"]
            
            # Truncate if too long
            if len(html_code) > HTML_MAX_LENGTH:
                html_code = html_code[:HTML_MAX_LENGTH] + "... [truncated]"
            
            result["html_code"] = html_code
        
        return result

    def analyze_for_segments(self, image_path: str) -> Dict:
        """
        Analyze an image for segmentation/detection of objects.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing detected objects and segmentation info
        """
        prompt = """Identify all objects in this image that could be segmented.
        List each object with its approximate location and confidence.
        Format: object_name, location (x,y,width,height), confidence"""

        result = self.analyze_image(image_path, prompt)
        
        if result["success"]:
            # Parse detected objects from text
            objects = []
            lines = result["text"].split("\n")
            for line in lines:
                if line.strip():
                    objects.append({"description": line.strip()})
            result["objects"] = objects
        
        return result


# Convenience function for quick testing
def create_client() -> GemmaClient:
    """Create a GemmaClient instance with default settings."""
    return GemmaClient()
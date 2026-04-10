"""
GemmaClient — Unified client for Gemma 4 via OpenRouter or Google AI.

Provider is selected automatically from the environment:
  OPENROUTER_API_KEY → uses OpenRouter  (OpenAI-compatible, recommended)
  GEMINI_API_KEY     → uses Google AI   (generativelanguage.googleapis.com)
  neither            → mock mode (realistic demo responses, no API calls)
"""

import base64
import json
import os
import time
from typing import Dict, List, Optional

from config import (
    API_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
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
    """Gemma 4 client — routes requests to OpenRouter or Google AI based on available keys."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = API_PROVIDER

        if self.provider == "openrouter":
            # Use explicit api_key if provided (even ""), otherwise fall back to config
            self.api_key = OPENROUTER_API_KEY if api_key is None else api_key
            self.model = model or OPENROUTER_MODEL
            self.base_url = OPENROUTER_BASE_URL
        elif self.provider == "google":
            self.api_key = GEMINI_API_KEY if api_key is None else api_key
            self.model = model or GEMINI_MODEL
            self.base_url = GEMINI_BASE_URL
        else:
            self.api_key = "" if api_key is None else api_key
            self.model = ""
            self.base_url = ""

        self.mock_mode = MOCK_MODE or not self.api_key

    # ── Image helpers ─────────────────────────────────────────────────────────

    def _encode_image(self, image_path: str) -> str:
        """Base64-encode an image file."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: str) -> str:
        """Return MIME type based on file extension."""
        ext = image_path.lower().rsplit(".", 1)[-1]
        return {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif", "webp": "image/webp",
        }.get(ext, "image/jpeg")

    # ── API dispatch ──────────────────────────────────────────────────────────

    def _call_api(self, prompt: str, image_path: Optional[str] = None) -> Optional[str]:
        """Call the configured provider. Returns response text, or None on failure."""
        if self.mock_mode:
            return self._mock_text(prompt)
        if self.provider == "openrouter":
            return self._call_openrouter(prompt, image_path)
        return self._call_google(prompt, image_path)

    def _call_openrouter(self, prompt: str, image_path: Optional[str] = None) -> Optional[str]:
        """OpenAI-compatible chat completions via OpenRouter."""
        import urllib.request

        content: List = [{"type": "text", "text": prompt}]
        if image_path:
            mime = self._get_mime_type(image_path)
            b64 = self._encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode(), headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"OpenRouter API failed: {e}")
                    return None
                time.sleep(RETRY_DELAY)
        return None

    def _call_google(self, prompt: str, image_path: Optional[str] = None) -> Optional[str]:
        """Google AI (generativelanguage.googleapis.com) Gemini-format request."""
        import urllib.request

        parts = [{"text": prompt}]
        if image_path:
            mime = self._get_mime_type(image_path)
            b64 = self._encode_image(image_path)
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
        }
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode(), headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode())
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Google AI API failed: {e}")
                    return None
                time.sleep(RETRY_DELAY)
        return None

    # ── Mock ──────────────────────────────────────────────────────────────────

    def _mock_text(self, prompt: str) -> str:
        """Return a realistic demo response without making any API call."""
        p = prompt.lower()

        if "element_type" in p or ("json" in p and "detected_elements" in p):
            return json.dumps({
                "description": "A modern analytics dashboard with a clean, professional design featuring data visualisation components.",
                "detected_elements": [
                    {"element_type": "navigation", "description": "Top nav bar with logo and links", "text_content": "Dashboard · Reports · Settings · Profile", "confidence": 0.95},
                    {"element_type": "header",     "description": "Page title with date range picker", "text_content": "Analytics Overview — Last 30 Days", "confidence": 0.93},
                    {"element_type": "card",        "description": "KPI card — total users",  "text_content": "12,543 Users ↑ 8.2%",   "confidence": 0.94},
                    {"element_type": "card",        "description": "KPI card — revenue",       "text_content": "$48,290 Revenue ↑ 12.1%", "confidence": 0.91},
                    {"element_type": "chart",       "description": "Line chart — daily active users", "text_content": None, "confidence": 0.89},
                    {"element_type": "table",       "description": "Recent events table",      "text_content": "Event | User | Date | Status", "confidence": 0.87},
                    {"element_type": "button",      "description": "Export CSV button",        "text_content": "Export CSV",            "confidence": 0.92},
                    {"element_type": "input",       "description": "Search / filter field",    "text_content": "Search events...",      "confidence": 0.88},
                ],
                "categories": ["dashboard", "analytics", "web-ui", "data-visualization", "admin-panel"],
                "text_content": "Dashboard · Reports · Settings · Analytics Overview · Last 30 Days · 12,543 Users · $48,290 Revenue · Export CSV · Search events...",
            })

        if "html" in p or "code" in p or "website" in p:
            return """\
<html>
<head>
  <title>Mock Website</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    .header { background: #333; color: white; padding: 16px 20px; border-radius: 6px; }
    .content { padding: 24px 0; }
    .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
  </style>
</head>
<body>
  <div class="header"><h1>Mock Header</h1></div>
  <div class="content">
    <div class="card">
      <h2>Welcome</h2>
      <p>This is a mock response — running without an API key.</p>
    </div>
  </div>
</body>
</html>"""

        if "segment" in p or "object" in p or "detect" in p:
            return (
                "Detected objects: person (0.95), car (0.88), building (0.76). "
                "Set OPENROUTER_API_KEY or GEMINI_API_KEY for real detection."
            )

        return (
            "Mock analysis: the image contains visual content. "
            "For real AI-powered analysis, add OPENROUTER_API_KEY (recommended) "
            "or GEMINI_API_KEY to your .env file. Currently running in mock mode."
        )

    # ── Public methods ────────────────────────────────────────────────────────

    def analyze_image(self, image_path: str, prompt: str = "Describe this image") -> Dict:
        """
        Analyse an image and return a text description.

        Args:
            image_path: Path to the image file
            prompt: Natural-language question or instruction

        Returns:
            {"success": bool, "text": str|None, "mock_mode": bool, "error": str|None}
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        text = self._call_api(prompt, image_path)
        if text is not None:
            return {"success": True, "text": text, "mock_mode": self.mock_mode}
        return {
            "success": False,
            "text": None,
            "mock_mode": self.mock_mode,
            "error": "API call failed — check your API key and network connection.",
        }

    def generate_html_from_screenshot(self, image_path: str) -> Dict:
        """
        Generate HTML/CSS code from a screenshot image.

        Args:
            image_path: Path to the screenshot

        Returns:
            {"success": bool, "html_code": str|None, "mock_mode": bool}
        """
        import re

        prompt = (
            "Convert this screenshot into clean, semantic HTML with inline CSS. "
            "Include proper <html>, <head>, and <body> tags. "
            "Output ONLY the HTML — no explanations, no markdown code blocks."
        )
        result = self.analyze_image(image_path, prompt)

        if result["success"]:
            html_code = result.get("text") or ""
            # Strip markdown code fences if the model wrapped its output
            html_code = re.sub(r"```(?:html)?\s*", "", html_code)
            html_code = re.sub(r"```\s*$", "", html_code).strip()
            # Try to extract a proper <html>…</html> block
            m = re.search(HTML_EXTRACTION_PATTERN, html_code, re.DOTALL | re.IGNORECASE)
            if m:
                html_code = m.group(0)
            if len(html_code) > HTML_MAX_LENGTH:
                html_code = html_code[:HTML_MAX_LENGTH] + "\n<!-- [truncated] -->"
            result["html_code"] = html_code

        return result

    def analyze_for_segments(self, image_path: str) -> Dict:
        """
        Ask the model to list every detectable object in the image.

        Args:
            image_path: Path to the image file

        Returns:
            {"success": bool, "objects": list, "text": str, "mock_mode": bool}
        """
        prompt = (
            "List every distinct object visible in this image. "
            "Output ONLY the objects, one per line, in exactly this format: "
            "name, location (top-left|top-center|top-right|middle-left|middle-center|middle-right|bottom-left|bottom-center|bottom-right), confidence\n"
            "Example:\n"
            "red rectangle, top-center, 0.95\n"
            "blue button, bottom-left, 0.88\n"
            "No introductory text. No bullet points. No numbering. Only the list."
        )
        result = self.analyze_image(image_path, prompt)
        if result["success"]:
            objects = []
            for line in (result.get("text") or "").split("\n"):
                line = line.strip().lstrip("*-•0123456789.) ")
                # Only keep lines that look like "name, location, confidence"
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and line:
                    objects.append({"description": line})
            result["objects"] = objects
        return result


# Convenience helper
def create_client() -> GemmaClient:
    """Create a GemmaClient with default settings."""
    return GemmaClient()

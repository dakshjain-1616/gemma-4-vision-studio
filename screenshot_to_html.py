"""
ScreenshotToHTML - Convert screenshots to HTML code using Gemma 4.
Calls GemmaClient for image analysis and extracts HTML code from responses.
"""

import re
from typing import Dict, Optional

from config import HTML_MAX_LENGTH, HTML_EXTRACTION_PATTERN
from gemma_client import GemmaClient


class ScreenshotToHTML:
    """Convert screenshots to HTML code using Gemma 4 vision model."""

    def __init__(self, client: Optional[GemmaClient] = None):
        """
        Initialize ScreenshotToHTML converter.

        Args:
            client: GemmaClient instance (optional, creates new if not provided)
        """
        self.client = client or GemmaClient()

    def convert(self, image_path: str) -> Dict:
        """
        Convert a screenshot to HTML code.

        Args:
            image_path: Path to the screenshot image

        Returns:
            Dictionary containing HTML code and metadata
        """
        # Call GemmaClient to generate HTML from screenshot
        result = self.client.generate_html_from_screenshot(image_path)
        
        if result.get("success"):
            html_code = result.get("html_code", "")
            
            # Validate HTML structure
            is_valid = self._validate_html(html_code)
            
            return {
                "success": True,
                "html_code": html_code,
                "is_valid": is_valid,
                "mock_mode": result.get("mock_mode", False),
                "length": len(html_code)
            }
        
        return {
            "success": False,
            "error": result.get("text", "Failed to generate HTML"),
            "mock_mode": result.get("mock_mode", False)
        }

    def _validate_html(self, html_code: str) -> bool:
        """
        Validate that the generated code has proper HTML structure.

        Args:
            html_code: HTML code string

        Returns:
            True if valid HTML structure, False otherwise
        """
        # Check for basic HTML tags
        has_html = bool(re.search(r"<html", html_code, re.IGNORECASE))
        has_head = bool(re.search(r"<head", html_code, re.IGNORECASE))
        has_body = bool(re.search(r"<body", html_code, re.IGNORECASE))

        return has_html and has_head and has_body

    def extract_html(self, text: str) -> Optional[str]:
        """
        Extract HTML code from a text response.

        Args:
            text: Text containing HTML code

        Returns:
            Extracted HTML code or None if not found
        """
        # Try to find HTML tags
        html_match = re.search(HTML_EXTRACTION_PATTERN, text, re.DOTALL | re.IGNORECASE)
        if html_match:
            return html_match.group(0)
        
        # Try to find code block
        code_match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Return full text if it looks like HTML
        if "<html" in text.lower() or "<div" in text.lower():
            return text
        
        return None


# Convenience function
def convert_screenshot(image_path: str) -> Dict:
    """
    Convert a screenshot to HTML code.

    Args:
        image_path: Path to the screenshot image

    Returns:
        Dictionary containing conversion results
    """
    converter = ScreenshotToHTML()
    return converter.convert(image_path)

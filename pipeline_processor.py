"""
Pipeline processor — integrates screenshot-to-LLM structured analysis with Gemma 4.
Produces: description, detected UI elements (type + confidence), categories, extracted text, Q&A.
"""

import json
import re
from typing import Optional, Dict, Any, List

from gemma_client import GemmaClient


class PipelineProcessor:
    """Structured image analysis pipeline powered by Gemma 4."""

    def __init__(self, client: Optional[GemmaClient] = None):
        self.client = client or GemmaClient()

    def analyze_structured(self, image_path: str, question: Optional[str] = None) -> Dict:
        """
        Run full structured analysis on an image.

        Returns description, detected elements, categories, extracted text,
        and an optional answer to the supplied question.
        """
        prompt = (
            "Analyze this image and return ONLY a valid JSON object with this exact structure:\n"
            "{\n"
            '  "description": "comprehensive description of the image",\n'
            '  "detected_elements": [\n'
            "    {\n"
            '      "element_type": "button|text|image|chart|table|form|navigation|header|footer|icon|input|card|menu|other",\n'
            '      "description": "what this element is and does",\n'
            '      "text_content": "visible text inside the element, or null",\n'
            '      "confidence": 0.95\n'
            "    }\n"
            "  ],\n"
            '  "categories": ["tag1", "tag2"],\n'
            '  "text_content": "all visible text in the image"\n'
            "}\n\n"
            "Return ONLY valid JSON. No markdown code blocks. No explanation outside the JSON."
        )

        result = self.client.analyze_image(image_path, prompt)

        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Analysis failed"),
                "mock_mode": result.get("mock_mode", True),
            }

        structured = self._parse_structured(result.get("text", ""))

        answer = None
        if question:
            answer = self._answer_question(image_path, structured, question)

        return {
            "success": True,
            "description": structured.get("description", ""),
            "detected_elements": structured.get("detected_elements", []),
            "categories": structured.get("categories", []),
            "text_content": structured.get("text_content", ""),
            "question": question,
            "answer": answer,
            "mock_mode": result.get("mock_mode", True),
        }

    # ── private helpers ──────────────────────────────────────────────────────

    def _parse_structured(self, text: str) -> Dict:
        """Parse JSON from model response, with graceful fallback."""
        # Strip markdown code fences
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting the outermost JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: treat the raw text as the description
        return {
            "description": text or "Analysis complete.",
            "detected_elements": [],
            "categories": ["image"],
            "text_content": "",
        }

    def _answer_question(self, image_path: str, analysis: Dict, question: str) -> str:
        """Answer a natural-language question about the image."""
        desc = analysis.get("description", "")
        elements = analysis.get("detected_elements", [])
        text_content = analysis.get("text_content", "")

        ctx_parts = [f"Image description: {desc}"]
        if elements:
            elem_lines = [
                f"{e.get('element_type','element')}: {e.get('description','')}"
                for e in elements[:8]
            ]
            ctx_parts.append("Elements: " + "; ".join(elem_lines))
        if text_content:
            ctx_parts.append(f"Text in image: {text_content[:400]}")

        prompt = (
            "You are analyzing an image. Answer the question concisely and directly based on what you see.\n\n"
            + "\n".join(ctx_parts)
            + f"\n\nQuestion: {question}\n\nAnswer:"
        )

        result = self.client.analyze_image(image_path, prompt)
        if result.get("success"):
            return result.get("text", "Could not answer the question.")
        return "Could not answer the question."

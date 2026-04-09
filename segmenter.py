"""
ImageSegmenter - Image segmentation using DETR model.
Provides object detection and segmentation with visual annotations.
Includes fallback to GemmaClient for analysis when DETR is unavailable.
"""

import os
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any

from PIL import Image, ImageDraw, ImageFont

from config import SEGMENTATION_MODEL, SEGMENTATION_THRESHOLD


class ImageSegmenter:
    """Image segmentation using DETR (Detection Transformer) model."""

    def __init__(self, model_name: Optional[str] = None, threshold: float = None):
        """
        Initialize the ImageSegmenter.

        Args:
            model_name: DETR model name (optional, uses default if not provided)
            threshold: Confidence threshold for detection (optional, uses default if not provided)
        """
        self.model_name = model_name or SEGMENTATION_MODEL
        self.threshold = threshold or SEGMENTATION_THRESHOLD
        self.model = None
        self.processor = None
        self.use_gemma_fallback = False
        
        # Try to load DETR model
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.use_gemma_fallback = False
        except Exception as e:
            print(f"DETR model loading failed: {e}")
            self.use_gemma_fallback = True

    def detect_objects(self, image_path: str) -> Dict:
        """
        Detect objects in an image using DETR.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing detected objects with boxes and labels
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if self.use_gemma_fallback:
            return self._gemma_fallback_detection(image_path)

        try:
            import torch
            
            image = Image.open(image_path)
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Run detection
            outputs = self.model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.threshold
            )[0]
            
            # Format results
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detections.append({
                    "label": self.model.config.id2label[label.item()],
                    "confidence": score.item(),
                    "box": box.tolist()  # [x_min, y_min, x_max, y_max]
                })
            
            return {
                "success": True,
                "detections": detections,
                "model_used": "DETR",
                "threshold": self.threshold
            }
            
        except Exception as e:
            print(f"Detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections": []
            }

    def _gemma_fallback_detection(self, image_path: str) -> Dict:
        """
        Fallback to GemmaClient for object detection when DETR is unavailable.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing detected objects from Gemma analysis
        """
        from gemma_client import GemmaClient
        
        client = GemmaClient()
        result = client.analyze_for_segments(image_path)
        
        detections = []
        if "objects" in result:
            for obj in result["objects"]:
                detections.append({
                    "label": obj.get("description", "unknown"),
                    "confidence": 0.5,  # Default confidence for fallback
                    "box": [0, 0, 100, 100]  # Placeholder box
                })
        
        return {
            "success": result.get("success", False),
            "detections": detections,
            "model_used": "GemmaClient (fallback)",
            "threshold": self.threshold
        }

    def draw_annotations(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Draw detection annotations on an image.

        Args:
            image_path: Path to the input image
            output_path: Path for output image (optional, returns in memory if not provided)

        Returns:
            Dictionary containing annotated image path/base64 and detection info
        """
        # Detect objects first
        detection_result = self.detect_objects(image_path)
        
        if not detection_result.get("success"):
            return {
                "success": False,
                "error": "Detection failed",
                "detections": []
            }
        
        # Load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw boxes and labels
        detections = detection_result.get("detections", [])
        for det in detections:
            box = det.get("box", [0, 0, 100, 100])
            label = det.get("label", "unknown")
            confidence = det.get("confidence", 0)
            
            # Draw rectangle
            draw.rectangle(box, outline="red", width=2)
            
            # Draw label
            text = f"{label}: {confidence:.2f}"
            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((box[0], box[1]), text, fill="white", font=font)
        
        # Save or return image
        if output_path:
            image.save(output_path)
            return {
                "success": True,
                "output_path": output_path,
                "detections": detections,
                "model_used": detection_result.get("model_used", "DETR")
            }
        else:
            # Return as base64
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            
            return {
                "success": True,
                "image_base64": image_base64,
                "detections": detections,
                "model_used": detection_result.get("model_used", "DETR")
            }

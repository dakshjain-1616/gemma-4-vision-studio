"""
FastAPI application for Gemma 4 Vision Studio.
Provides endpoints for image analysis, segmentation, and screenshot-to-HTML conversion.
"""

import os
import base64
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    APP_HOST,
    APP_PORT,
    CORS_ORIGINS,
    MAX_IMAGE_SIZE_MB,
    ALLOWED_FORMATS,
    MOCK_MODE,
)
from gemma_client import GemmaClient
from segmenter import ImageSegmenter
from screenshot_to_html import ScreenshotToHTML
from pipeline_processor import PipelineProcessor


# Initialize FastAPI app
app = FastAPI(
    title="Gemma 4 Vision Studio",
    description="Image analysis, segmentation, and screenshot-to-HTML powered by Gemma 4",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Initialize clients
gemma_client = GemmaClient()
segmenter = ImageSegmenter()
screenshot_to_html = ScreenshotToHTML()
pipeline_processor = PipelineProcessor(client=gemma_client)


class HealthResponse(BaseModel):
    status: str
    mock_mode: bool
    version: str


class AnalysisResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    mock_mode: bool
    error: Optional[str] = None


class SegmentationResponse(BaseModel):
    success: bool
    detections: list
    image_base64: Optional[str] = None
    mock_mode: bool
    error: Optional[str] = None


class HTMLResponse(BaseModel):
    success: bool
    html_code: Optional[str] = None
    is_valid: bool
    mock_mode: bool
    error: Optional[str] = None


class PipelineResponse(BaseModel):
    success: bool
    description: Optional[str] = None
    detected_elements: list = []
    categories: list = []
    text_content: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    mock_mode: bool
    error: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "version": "1.0.0"
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = "Describe this image"
):
    """
    Analyze an image using Gemma 4 vision model.
    
    Args:
        file: Uploaded image file
        prompt: Analysis prompt (optional)
    
    Returns:
        Analysis results
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check format
    ext = file.filename.lower().split(".")[-1]
    if ext not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Invalid format. Allowed: {ALLOWED_FORMATS}")
    
    # Read file
    content = await file.read()
    
    # Check size
    if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_IMAGE_SIZE_MB}MB")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Analyze image
        result = gemma_client.analyze_image(tmp_path, prompt)
        
        return {
            "success": result.get("success", False),
            "text": result.get("text", ""),
            "mock_mode": result.get("mock_mode", MOCK_MODE),
            "error": result.get("error")
        }
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(file: UploadFile = File(...)):
    """
    Segment/detect objects in an image.
    
    Args:
        file: Uploaded image file
    
    Returns:
        Detection results with annotated image
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check format
    ext = file.filename.lower().split(".")[-1]
    if ext not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Invalid format. Allowed: {ALLOWED_FORMATS}")
    
    # Read file
    content = await file.read()
    
    # Check size
    if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_IMAGE_SIZE_MB}MB")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Draw annotations
        result = segmenter.draw_annotations(tmp_path)
        
        return {
            "success": result.get("success", False),
            "detections": result.get("detections", []),
            "image_base64": result.get("image_base64"),
            "mock_mode": MOCK_MODE,
            "error": result.get("error")
        }
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.post("/screenshot-to-html", response_model=HTMLResponse)
async def screenshot_to_html_endpoint(file: UploadFile = File(...)):
    """
    Convert a screenshot to HTML code.
    
    Args:
        file: Uploaded screenshot image
    
    Returns:
        Generated HTML code
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check format
    ext = file.filename.lower().split(".")[-1]
    if ext not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Invalid format. Allowed: {ALLOWED_FORMATS}")
    
    # Read file
    content = await file.read()
    
    # Check size
    if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_IMAGE_SIZE_MB}MB")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Convert screenshot
        result = screenshot_to_html.convert(tmp_path)
        
        return {
            "success": result.get("success", False),
            "html_code": result.get("html_code"),
            "is_valid": result.get("is_valid", False),
            "mock_mode": result.get("mock_mode", MOCK_MODE),
            "error": result.get("error")
        }
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline_endpoint(
    file: UploadFile = File(...),
    question: Optional[str] = None,
):
    """
    Structured analysis pipeline: description, detected elements, categories,
    extracted text, and optional Q&A powered by Gemma 4.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = file.filename.lower().split(".")[-1]
    if ext not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Invalid format. Allowed: {ALLOWED_FORMATS}")

    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_IMAGE_SIZE_MB}MB")

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline_processor.analyze_structured(tmp_path, question=question or None)
        return {
            "success": result.get("success", False),
            "description": result.get("description"),
            "detected_elements": result.get("detected_elements", []),
            "categories": result.get("categories", []),
            "text_content": result.get("text_content"),
            "question": result.get("question"),
            "answer": result.get("answer"),
            "mock_mode": result.get("mock_mode", MOCK_MODE),
            "error": result.get("error"),
        }
    finally:
        os.unlink(tmp_path)


@app.get("/")
async def root():
    """Root endpoint - serves the main HTML page."""
    return FileResponse(os.path.join(static_path, "index.html"))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "mock_mode": MOCK_MODE}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "mock_mode": MOCK_MODE}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

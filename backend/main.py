"""
FastAPI backend for Deepfake Video Detector
Provides REST API endpoints for video analysis and deepfake detection
From Hasif's Workspace
"""

import os
import sys
import tempfile
import uuid
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model_handler import ModelHandler
from video_processor import VideoProcessor
from explainability_engine import ExplainabilityEngine
from config import get_settings

# Initialize settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Video Detector API",
    description="AI-powered deepfake video detection with explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_handler = ModelHandler()
video_processor = VideoProcessor()
explainability_engine = ExplainabilityEngine()


# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    num_frames: int = 5
    enable_gradcam: bool = True
    confidence_threshold: float = 0.5


class FramePrediction(BaseModel):
    frame_number: int
    prediction: str
    confidence: float
    gradcam_available: bool = False
    gradcam_path: Optional[str] = None


class AnalysisResponse(BaseModel):
    video_id: str
    overall_prediction: str
    confidence_score: float
    frame_predictions: List[FramePrediction]
    processing_time: float
    model_version: str
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime: float


# Global variables for tracking
start_time = None


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global start_time
    import time

    start_time = time.time()

    # Load model
    await model_handler.load_model()

    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.MODEL_PATH, exist_ok=True)

    print("🚀 Deepfake Detector API started successfully!")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Deepfake Video Detector API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import time

    uptime = time.time() - start_time if start_time else 0

    return HealthResponse(
        status="healthy",
        model_loaded=model_handler.is_loaded(),
        version="1.0.0",
        uptime=uptime,
    )


@app.post("/api/v1/analyze-video", response_model=AnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    num_frames: int = Form(5),
    enable_gradcam: bool = Form(True),
    confidence_threshold: float = Form(0.5),
):
    """
    Analyze uploaded video for deepfake detection

    Args:
        video_file: Video file to analyze (MP4, AVI, MOV)
        num_frames: Number of frames to extract and analyze
        enable_gradcam: Whether to generate Grad-CAM visualizations
        confidence_threshold: Confidence threshold for classification

    Returns:
        Analysis results with predictions and explanations
    """
    import time

    start_time = time.time()

    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    file_extension = Path(video_file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Generate unique video ID
    video_id = str(uuid.uuid4())

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as tmp_file:
            content = await video_file.read()
            tmp_file.write(content)
            temp_video_path = tmp_file.name

        # Extract frames from video
        frames = await video_processor.extract_frames(
            temp_video_path, num_frames=num_frames
        )

        if not frames:
            raise HTTPException(
                status_code=400, detail="Could not extract frames from video"
            )

        # Process frames through model
        frame_predictions = []
        all_confidences = []

        for i, frame in enumerate(frames):
            # Get prediction for frame
            prediction, confidence = await model_handler.predict_frame(frame)

            frame_pred = FramePrediction(
                frame_number=i + 1,
                prediction=prediction,
                confidence=confidence,
                gradcam_available=enable_gradcam,
            )

            # Generate Grad-CAM if requested
            if enable_gradcam:
                try:
                    gradcam_path = await explainability_engine.generate_gradcam(
                        frame, model_handler.model, video_id, i + 1
                    )
                    frame_pred.gradcam_path = gradcam_path
                except Exception as e:
                    print(f"Grad-CAM generation failed for frame {i + 1}: {e}")

            frame_predictions.append(frame_pred)
            all_confidences.append(confidence)

        # Calculate overall prediction
        avg_confidence = sum(all_confidences) / len(all_confidences)
        overall_prediction = (
            "Deepfake" if avg_confidence > confidence_threshold else "Real"
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Prepare metadata
        metadata = {
            "original_filename": video_file.filename,
            "file_size": len(content),
            "frames_extracted": len(frames),
            "video_duration": await video_processor.get_video_duration(temp_video_path),
            "model_architecture": "EfficientNet-B0",
            "confidence_threshold": confidence_threshold,
        }

        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_temp_files, temp_video_path)

        return AnalysisResponse(
            video_id=video_id,
            overall_prediction=overall_prediction,
            confidence_score=avg_confidence,
            frame_predictions=frame_predictions,
            processing_time=processing_time,
            model_version=model_handler.get_model_version(),
            metadata=metadata,
        )

    except Exception as e:
        # Cleanup on error
        if "temp_video_path" in locals():
            try:
                os.unlink(temp_video_path)
            except:
                pass

        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/api/v1/gradcam/{video_id}/{frame_number}")
async def get_gradcam_image(video_id: str, frame_number: int):
    """
    Retrieve Grad-CAM visualization for a specific frame

    Args:
        video_id: Unique video identifier
        frame_number: Frame number to retrieve

    Returns:
        Grad-CAM visualization image
    """
    gradcam_path = os.path.join(
        settings.OUTPUT_DIR, f"gradcam_{video_id}_frame_{frame_number}.png"
    )

    if not os.path.exists(gradcam_path):
        raise HTTPException(status_code=404, detail="Grad-CAM visualization not found")

    return FileResponse(
        gradcam_path,
        media_type="image/png",
        filename=f"gradcam_frame_{frame_number}.png",
    )


@app.get("/api/v1/models/info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_name": "DeepfakeDetector",
        "architecture": "EfficientNet-B0",
        "version": model_handler.get_model_version(),
        "loaded": model_handler.is_loaded(),
        "input_size": [224, 224],
        "classes": ["Real", "Deepfake"],
    }


async def cleanup_temp_files(file_path: str):
    """Background task to cleanup temporary files"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error cleaning up temporary file {file_path}: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

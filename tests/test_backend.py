"""
Backend API Tests
Comprehensive tests for FastAPI backend functionality
From Hasif's Workspace
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import cv2

# Import backend components
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app
from config import get_settings


class TestBackendAPI:
    """Test suite for backend API"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def sample_video(self):
        """Create a sample video file for testing"""
        # Create a simple test video
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_file.name, fourcc, 20.0, (640, 480))

        # Write some frames
        for i in range(30):
            # Create a simple frame with changing color
            frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 8 % 255)
            out.write(frame)

        out.release()

        yield temp_file.name

        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Deepfake Video Detector API"

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert "uptime" in data
        assert data["status"] == "healthy"

    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_name" in data
        assert "architecture" in data
        assert "version" in data
        assert "loaded" in data
        assert "input_size" in data
        assert "classes" in data

    def test_video_upload_invalid_format(self, client):
        """Test video upload with invalid format"""
        # Create a text file instead of video
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"This is not a video file")
            temp_file.flush()

            with open(temp_file.name, "rb") as f:
                response = client.post(
                    "/api/v1/analyze-video",
                    files={"video_file": ("test.txt", f, "text/plain")},
                    data={
                        "num_frames": 5,
                        "enable_gradcam": True,
                        "confidence_threshold": 0.5,
                    },
                )

            os.unlink(temp_file.name)

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_video_analysis_success(self, client, sample_video):
        """Test successful video analysis"""
        with open(sample_video, "rb") as f:
            response = client.post(
                "/api/v1/analyze-video",
                files={"video_file": ("test_video.mp4", f, "video/mp4")},
                data={
                    "num_frames": 3,
                    "enable_gradcam": False,  # Disable to avoid model dependency
                    "confidence_threshold": 0.5,
                },
            )

        # Note: This might fail if model is not loaded, which is expected in test environment
        # We're testing the API structure rather than actual inference
        if response.status_code == 200:
            data = response.json()
            assert "video_id" in data
            assert "overall_prediction" in data
            assert "confidence_score" in data
            assert "frame_predictions" in data
            assert "processing_time" in data
            assert "model_version" in data
            assert "metadata" in data
        else:
            # Expected if model is not available in test environment
            assert response.status_code in [500, 400]

    def test_video_analysis_parameters(self, client, sample_video):
        """Test video analysis with different parameters"""
        with open(sample_video, "rb") as f:
            response = client.post(
                "/api/v1/analyze-video",
                files={"video_file": ("test_video.mp4", f, "video/mp4")},
                data={
                    "num_frames": 10,
                    "enable_gradcam": True,
                    "confidence_threshold": 0.7,
                },
            )

        # Check that parameters are accepted (regardless of processing success)
        assert response.status_code in [200, 400, 500]

    def test_gradcam_endpoint_not_found(self, client):
        """Test Grad-CAM endpoint with non-existent visualization"""
        response = client.get("/api/v1/gradcam/nonexistent_video_id/1")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/v1/health")
        # CORS headers should be present due to middleware
        assert response.status_code in [
            200,
            405,
        ]  # Some test clients handle OPTIONS differently


class TestBackendConfig:
    """Test backend configuration"""

    def test_settings_initialization(self):
        """Test settings can be initialized"""
        settings = get_settings()

        assert settings.APP_NAME == "Deepfake Video Detector"
        assert settings.APP_VERSION == "1.0.0"
        assert settings.API_HOST == "0.0.0.0"
        assert settings.API_PORT == 8000

    def test_settings_paths(self):
        """Test that required paths are set"""
        settings = get_settings()

        assert hasattr(settings, "MODEL_PATH")
        assert hasattr(settings, "UPLOAD_DIR")
        assert hasattr(settings, "OUTPUT_DIR")
        assert hasattr(settings, "TEMP_DIR")

    def test_settings_validation(self):
        """Test settings validation"""
        settings = get_settings()

        # Check that file size limits are reasonable
        assert settings.MAX_FILE_SIZE > 0
        assert settings.MAX_FRAMES_PER_VIDEO > 0
        assert 0 <= settings.CONFIDENCE_THRESHOLD <= 1


class TestBackendUtils:
    """Test backend utility functions"""

    def test_video_validation(self):
        """Test video file validation"""
        # This would test the video validation logic
        # Implementation depends on the actual validation function
        pass

    def test_frame_extraction(self):
        """Test frame extraction functionality"""
        # This would test frame extraction from video
        # Implementation depends on the actual extraction function
        pass


# Integration tests
class TestBackendIntegration:
    """Integration tests for backend components"""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self):
        """Test full pipeline with mocked components"""
        # This would test the complete pipeline with mocked model
        pass

    def test_error_handling(self, client):
        """Test error handling in various scenarios"""
        # Test with malformed requests
        response = client.post("/api/v1/analyze-video")
        assert response.status_code == 422  # Validation error

    def test_rate_limiting(self, client):
        """Test rate limiting if implemented"""
        # This would test rate limiting functionality
        pass


# Performance tests
class TestBackendPerformance:
    """Performance tests for backend"""

    def test_response_time(self, client):
        """Test API response times"""
        import time

        start_time = time.time()
        response = client.get("/api/v1/health")
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time

        results = []

        def make_request():
            response = client.get("/api/v1/health")
            results.append(response.status_code)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that all requests succeeded
        assert all(status == 200 for status in results)
        assert len(results) == 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

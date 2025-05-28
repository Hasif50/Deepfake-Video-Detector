"""
Test package for Deepfake Video Detector
Comprehensive test suite for all components
From Hasif's Workspace
"""

__version__ = "1.0.0"

# Test configuration
TEST_CONFIG = {
    "data_dir": "./tests/data",
    "temp_dir": "./tests/temp",
    "output_dir": "./tests/output",
    "sample_video_path": "./tests/data/sample_video.mp4",
    "sample_frames_dir": "./tests/data/sample_frames",
    "model_path": "./tests/data/test_model.pth",
}


# Test utilities
def get_test_config():
    """Get test configuration"""
    return TEST_CONFIG.copy()


def setup_test_environment():
    """Setup test environment"""
    import os

    # Create test directories
    for key, path in TEST_CONFIG.items():
        if key.endswith("_dir"):
            os.makedirs(path, exist_ok=True)


def cleanup_test_environment():
    """Cleanup test environment"""
    import shutil
    import os

    # Remove test directories
    for key, path in TEST_CONFIG.items():
        if key.endswith("_dir") and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

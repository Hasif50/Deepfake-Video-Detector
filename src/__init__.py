"""
Deepfake Video Detector - Core Source Package
Enhanced modular architecture for deepfake detection
From Hasif's Workspace
"""

__version__ = "1.0.0"
__author__ = "Mohd Hasif"
__email__ = "hashifu50@gmail.com"
__workspace__ = "Hasif's Workspace"

# Package information
PACKAGE_NAME = "deepfake_detector"
DESCRIPTION = (
    "AI-powered deepfake video detection with explainable AI - From Hasif's Workspace"
)

# Import main components
try:
    from .models.deepfake_detector import DeepfakeDetector
    from .data.preprocessor import VideoPreprocessor
    from .training.trainer import ModelTrainer
    from .evaluation.evaluator import ModelEvaluator

    __all__ = [
        "DeepfakeDetector",
        "VideoPreprocessor",
        "ModelTrainer",
        "ModelEvaluator",
    ]

except ImportError as e:
    # Graceful handling of import errors during development
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []


# Version information
def get_version():
    """Get package version"""
    return __version__


def get_package_info():
    """Get package information"""
    return {
        "name": PACKAGE_NAME,
        "version": __version__,
        "description": DESCRIPTION,
        "author": __author__,
        "email": __email__,
    }

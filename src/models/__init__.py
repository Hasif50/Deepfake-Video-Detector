"""
Models package for deepfake detection
Contains model architectures and utilities
"""

from .deepfake_detector import DeepfakeDetector
from .model_utils import ModelUtils

__all__ = ["DeepfakeDetector", "ModelUtils"]

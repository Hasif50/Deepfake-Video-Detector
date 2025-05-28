"""
Training package for deepfake detection
Contains training utilities, trainers, and optimization components
"""

from .trainer import ModelTrainer
from .losses import DeepfakeLoss
from .metrics import DeepfakeMetrics

__all__ = ["ModelTrainer", "DeepfakeLoss", "DeepfakeMetrics"]

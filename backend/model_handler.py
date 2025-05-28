"""
Model Handler for Deepfake Detection
Manages model loading, inference, and prediction logic
From Hasif's Workspace
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torchvision import transforms
from PIL import Image
import asyncio
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from models.deepfake_detector import DeepfakeDetector
except ImportError:
    # Fallback implementation if main model not available
    import torchvision.models as models

    class DeepfakeDetector(nn.Module):
        def __init__(self, num_classes=1, pretrained=True):
            super(DeepfakeDetector, self).__init__()
            self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True), nn.Linear(num_ftrs, num_classes)
            )

        def forward(self, x):
            return self.efficientnet(x)


from config import get_settings

logger = logging.getLogger(__name__)


class ModelHandler:
    """Handles model loading, inference, and prediction operations"""

    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._create_transform()
        self._model_loaded = False
        self._model_version = "efficientnet_b0_v1.0"

        logger.info(f"ModelHandler initialized on device: {self.device}")

    def _create_transform(self) -> transforms.Compose:
        """Create image preprocessing transform pipeline"""
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    async def load_model(self) -> bool:
        """
        Load the deepfake detection model

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Initialize model
            self.model = DeepfakeDetector(num_classes=1, pretrained=True)

            # Try to load trained weights
            model_path = os.path.join(
                self.settings.MODEL_PATH, "deepfake_detector_best.pth"
            )

            if os.path.exists(model_path):
                logger.info(f"Loading trained model weights from {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("Trained model weights loaded successfully")
            else:
                logger.warning(f"Trained weights not found at {model_path}")
                logger.info("Using pre-trained EfficientNet-B0 weights only")

                # Create model directory if it doesn't exist
                os.makedirs(self.settings.MODEL_PATH, exist_ok=True)

            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()

            self._model_loaded = True
            logger.info("Model loaded and ready for inference")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model_loaded = False
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._model_loaded and self.model is not None

    def get_model_version(self) -> str:
        """Get model version string"""
        return self._model_version

    async def predict_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Predict if a single frame is real or deepfake

        Args:
            frame: Input frame as numpy array (H, W, C) in RGB format

        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Preprocess frame
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            # Apply transforms
            input_tensor = self.transform(frame).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).item()

            # Convert to prediction
            prediction = "Deepfake" if probability > 0.5 else "Real"
            confidence = probability if prediction == "Deepfake" else 1 - probability

            return prediction, confidence

        except Exception as e:
            logger.error(f"Error during frame prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    async def predict_batch(self, frames: list) -> list:
        """
        Predict multiple frames in batch for efficiency

        Args:
            frames: List of frames as numpy arrays

        Returns:
            List of (prediction, confidence) tuples
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Preprocess all frames
            batch_tensors = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                tensor = self.transform(frame)
                batch_tensors.append(tensor)

            # Stack into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Run batch inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()

            # Convert to predictions
            results = []
            for prob in probabilities:
                prediction = "Deepfake" if prob > 0.5 else "Real"
                confidence = prob if prediction == "Deepfake" else 1 - prob
                results.append((prediction, confidence))

            return results

        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
            raise RuntimeError(f"Batch prediction failed: {e}")

    def get_model_info(self) -> dict:
        """Get detailed model information"""
        return {
            "architecture": "EfficientNet-B0",
            "version": self._model_version,
            "device": str(self.device),
            "loaded": self._model_loaded,
            "input_size": [224, 224],
            "num_classes": 1,
            "output_type": "binary_classification",
            "preprocessing": {
                "resize": [224, 224],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        }

    def get_target_layer(self):
        """Get the target layer for Grad-CAM visualization"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        try:
            # For EfficientNet-B0, use the last convolutional block
            return self.model.efficientnet.features[-1]
        except AttributeError:
            # Fallback for different model structures
            logger.warning(
                "Could not find standard EfficientNet features, using fallback"
            )
            return None

    async def warmup(self, num_warmup_runs: int = 3):
        """
        Warm up the model with dummy inputs for consistent timing

        Args:
            num_warmup_runs: Number of warmup inference runs
        """
        if not self.is_loaded():
            logger.warning("Cannot warmup: model not loaded")
            return

        logger.info(f"Warming up model with {num_warmup_runs} runs...")

        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        with torch.no_grad():
            for i in range(num_warmup_runs):
                _ = self.model(dummy_input)

        logger.info("Model warmup completed")

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

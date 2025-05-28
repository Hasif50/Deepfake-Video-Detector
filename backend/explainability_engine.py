"""
Explainability Engine for Deepfake Detection
Generates Grad-CAM visualizations and other XAI explanations
From Hasif's Workspace
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import logging
from pathlib import Path

# Grad-CAM imports
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logging.warning("pytorch-grad-cam not available. Grad-CAM functionality disabled.")

from config import get_settings

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """Handles explainable AI functionality for deepfake detection"""

    def __init__(self):
        self.settings = get_settings()
        self.gradcam_available = GRADCAM_AVAILABLE

        # Create output directory for visualizations
        os.makedirs(self.settings.OUTPUT_DIR, exist_ok=True)

        logger.info(
            f"ExplainabilityEngine initialized. Grad-CAM available: {self.gradcam_available}"
        )

    async def generate_gradcam(
        self,
        frame: np.ndarray,
        model: torch.nn.Module,
        video_id: str,
        frame_number: int,
        target_category: Optional[int] = None,
    ) -> Optional[str]:
        """
        Generate Grad-CAM visualization for a frame

        Args:
            frame: Input frame as numpy array (RGB)
            model: PyTorch model for analysis
            video_id: Unique video identifier
            frame_number: Frame number in video
            target_category: Target class for visualization (None for predicted class)

        Returns:
            Path to generated Grad-CAM image or None if failed
        """
        if not self.gradcam_available:
            logger.warning("Grad-CAM not available")
            return None

        try:
            # Get target layer from model
            target_layer = self._get_target_layer(model)
            if target_layer is None:
                logger.error("Could not identify target layer for Grad-CAM")
                return None

            # Preprocess frame for model
            input_tensor = self._preprocess_frame(frame)
            if input_tensor is None:
                return None

            # Create Grad-CAM object
            cam = GradCAM(
                model=model, target_layers=[target_layer], use_cuda=input_tensor.is_cuda
            )

            # Define targets
            targets = None
            if target_category is not None:
                targets = [ClassifierOutputTarget(target_category)]

            # Generate Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Get the first (and only) image from batch
            grayscale_cam = grayscale_cam[0, :]

            # Create visualization
            visualization = self._create_visualization(frame, grayscale_cam)

            # Save visualization
            output_path = os.path.join(
                self.settings.OUTPUT_DIR, f"gradcam_{video_id}_frame_{frame_number}.png"
            )

            success = self._save_visualization(visualization, output_path)

            if success:
                logger.info(f"Grad-CAM saved to {output_path}")
                return output_path
            else:
                logger.error("Failed to save Grad-CAM visualization")
                return None

        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            return None

    def _get_target_layer(self, model: torch.nn.Module):
        """
        Identify the target layer for Grad-CAM visualization

        Args:
            model: PyTorch model

        Returns:
            Target layer or None if not found
        """
        try:
            # For EfficientNet-B0 in DeepfakeDetector
            if hasattr(model, "efficientnet") and hasattr(
                model.efficientnet, "features"
            ):
                return model.efficientnet.features[-1]

            # Fallback: look for last convolutional layer
            conv_layers = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_layers.append(module)

            if conv_layers:
                return conv_layers[-1]

            logger.warning("Could not find suitable target layer")
            return None

        except Exception as e:
            logger.error(f"Error finding target layer: {e}")
            return None

    def _preprocess_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess frame for model input

        Args:
            frame: Input frame as numpy array (RGB)

        Returns:
            Preprocessed tensor or None if failed
        """
        try:
            from torchvision import transforms

            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            # Create transform pipeline
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # Apply transforms and add batch dimension
            tensor = transform(frame).unsqueeze(0)

            # Move to appropriate device
            device = next(iter(torch.nn.Module.parameters(torch.nn.Module())))
            try:
                device = next(iter(torch.nn.Module.parameters(torch.nn.Module())))
                tensor = tensor.to(device)
            except:
                # Fallback to CPU
                tensor = tensor.to("cpu")

            return tensor

        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return None

    def _create_visualization(
        self, original_frame: np.ndarray, grayscale_cam: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Create Grad-CAM visualization overlay

        Args:
            original_frame: Original frame (RGB)
            grayscale_cam: Grad-CAM heatmap

        Returns:
            Visualization image or None if failed
        """
        try:
            # Normalize original frame to [0, 1]
            if original_frame.dtype == np.uint8:
                frame_normalized = original_frame.astype(np.float32) / 255.0
            else:
                frame_normalized = original_frame.astype(np.float32)
                if frame_normalized.max() > 1.0:
                    frame_normalized = frame_normalized / 255.0

            # Resize frame to match Grad-CAM if needed
            if frame_normalized.shape[:2] != grayscale_cam.shape:
                frame_normalized = cv2.resize(
                    frame_normalized, (grayscale_cam.shape[1], grayscale_cam.shape[0])
                )

            # Create overlay using show_cam_on_image
            visualization = show_cam_on_image(
                frame_normalized, grayscale_cam, use_rgb=True
            )

            return visualization

        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None

    def _save_visualization(self, visualization: np.ndarray, output_path: str) -> bool:
        """
        Save visualization to file

        Args:
            visualization: Visualization image
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save using matplotlib for better quality
            plt.figure(figsize=(10, 8))
            plt.imshow(visualization)
            plt.axis("off")
            plt.title("Grad-CAM Visualization", fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            return True

        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            return False

    async def generate_batch_gradcam(
        self,
        frames: list,
        model: torch.nn.Module,
        video_id: str,
        start_frame_number: int = 1,
    ) -> list:
        """
        Generate Grad-CAM for multiple frames

        Args:
            frames: List of frames as numpy arrays
            model: PyTorch model
            video_id: Unique video identifier
            start_frame_number: Starting frame number

        Returns:
            List of output paths (None for failed generations)
        """
        results = []

        for i, frame in enumerate(frames):
            frame_number = start_frame_number + i
            gradcam_path = await self.generate_gradcam(
                frame, model, video_id, frame_number
            )
            results.append(gradcam_path)

        return results

    def is_available(self) -> bool:
        """Check if Grad-CAM functionality is available"""
        return self.gradcam_available

    def get_supported_explanations(self) -> list:
        """Get list of supported explanation methods"""
        methods = []
        if self.gradcam_available:
            methods.append("grad_cam")
        return methods

    async def cleanup_old_visualizations(self, max_age_hours: int = 24):
        """
        Clean up old visualization files

        Args:
            max_age_hours: Maximum age of files to keep (in hours)
        """
        try:
            import time

            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            output_dir = Path(self.settings.OUTPUT_DIR)
            if not output_dir.exists():
                return

            deleted_count = 0
            for file_path in output_dir.glob("gradcam_*.png"):
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {file_path}: {e}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old visualization files")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

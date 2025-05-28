"""
Enhanced DeepfakeDetector Model
Advanced architecture with improved features and flexibility
From Hasif's Workspace
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DeepfakeDetector(nn.Module):
    """
    Enhanced Deepfake detection model using EfficientNet-B0 backbone
    with additional features and improved architecture
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        architecture: str = "efficientnet_b0",
    ):
        """
        Initialize the DeepfakeDetector model

        Args:
            num_classes: Number of output classes (1 for binary classification)
            pretrained: Whether to use pre-trained weights
            dropout_rate: Dropout rate for regularization
            architecture: Backbone architecture to use
        """
        super(DeepfakeDetector, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.architecture = architecture

        # Initialize backbone
        self._init_backbone(pretrained)

        # Initialize classifier
        self._init_classifier()

        # Model metadata
        self.model_info = {
            "architecture": architecture,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout_rate": dropout_rate,
            "input_size": (224, 224),
            "version": "1.0.0",
        }

        logger.info(f"DeepfakeDetector initialized with {architecture}")

    def _init_backbone(self, pretrained: bool):
        """Initialize the backbone network"""
        if self.architecture == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            # Remove the original classifier
            self.backbone.classifier = nn.Identity()

        elif self.architecture == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif self.architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

    def _init_classifier(self):
        """Initialize the classification head"""
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate / 2, inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate / 4, inplace=True),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Output tensor (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)

        # Apply classifier
        output = self.classifier(features)

        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features

    def get_gradcam_target_layer(self):
        """
        Get the target layer for Grad-CAM visualization

        Returns:
            Target layer for Grad-CAM
        """
        if self.architecture.startswith("efficientnet"):
            return self.backbone.features[-1]
        elif self.architecture == "resnet50":
            return self.backbone.layer4[-1]
        else:
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen for fine-tuning")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
        }


class EnsembleDeepfakeDetector(nn.Module):
    """
    Ensemble model combining multiple DeepfakeDetector models
    for improved accuracy and robustness
    """

    def __init__(self, models: list, weights: Optional[list] = None):
        """
        Initialize ensemble model

        Args:
            models: List of DeepfakeDetector models
            weights: Optional weights for each model
        """
        super(EnsembleDeepfakeDetector, self).__init__()

        self.models = nn.ModuleList(models)

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), (
                "Number of weights must match number of models"
            )
            self.weights = weights

        self.num_models = len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble

        Args:
            x: Input tensor

        Returns:
            Weighted average of model outputs
        """
        outputs = []

        for model in self.models:
            output = model(x)
            outputs.append(output)

        # Weighted average
        weighted_output = sum(w * out for w, out in zip(self.weights, outputs))

        return weighted_output

    def get_individual_predictions(self, x: torch.Tensor) -> list:
        """Get predictions from individual models"""
        predictions = []

        with torch.no_grad():
            for model in self.models:
                output = model(x)
                pred = torch.sigmoid(output)
                predictions.append(pred)

        return predictions


# Factory function for creating models
def create_deepfake_detector(
    architecture: str = "efficientnet_b0",
    num_classes: int = 1,
    pretrained: bool = True,
    **kwargs,
) -> DeepfakeDetector:
    """
    Factory function to create DeepfakeDetector models

    Args:
        architecture: Model architecture
        num_classes: Number of output classes
        pretrained: Use pre-trained weights
        **kwargs: Additional arguments

    Returns:
        DeepfakeDetector model
    """
    return DeepfakeDetector(
        num_classes=num_classes,
        pretrained=pretrained,
        architecture=architecture,
        **kwargs,
    )


# Model configurations
MODEL_CONFIGS = {
    "efficientnet_b0": {
        "architecture": "efficientnet_b0",
        "input_size": (224, 224),
        "params": "5.3M",
        "description": "Lightweight and efficient",
    },
    "efficientnet_b1": {
        "architecture": "efficientnet_b1",
        "input_size": (240, 240),
        "params": "7.8M",
        "description": "Better accuracy, slightly larger",
    },
    "resnet50": {
        "architecture": "resnet50",
        "input_size": (224, 224),
        "params": "25.6M",
        "description": "Classic architecture, robust",
    },
}


def get_available_models() -> Dict[str, Dict]:
    """Get available model configurations"""
    return MODEL_CONFIGS.copy()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing DeepfakeDetector...")

    # Create model
    model = create_deepfake_detector("efficientnet_b0")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model info: {model.get_model_info()}")
    print(f"Parameter count: {model.get_parameter_count()}")

    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Features shape: {features.shape}")

    print("DeepfakeDetector test completed successfully!")

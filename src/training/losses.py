"""
Advanced Loss Functions for Deepfake Detection
Comprehensive loss implementations with various strategies
From Hasif's Workspace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DeepfakeLoss(nn.Module):
    """
    Advanced loss function for deepfake detection
    Supports multiple loss types and combinations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DeepfakeLoss

        Args:
            config: Loss configuration dictionary
        """
        super(DeepfakeLoss, self).__init__()

        self.config = config or self._get_default_config()
        self.loss_type = self.config.get("type", "bce_with_logits")

        # Initialize loss components
        self._init_loss_functions()

        logger.info(f"Initialized {self.loss_type} loss function")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default loss configuration"""
        return {
            "type": "bce_with_logits",
            "pos_weight": 1.0,
            "label_smoothing": 0.0,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "dice_smooth": 1e-6,
            "combination_weights": {"bce": 0.7, "focal": 0.3},
        }

    def _init_loss_functions(self):
        """Initialize loss function components"""
        pos_weight = self.config.get("pos_weight", 1.0)

        if isinstance(pos_weight, (int, float)):
            pos_weight = torch.tensor([pos_weight])

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_loss_no_logits = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

        # Label smoothing parameter
        self.label_smoothing = self.config.get("label_smoothing", 0.0)

        # Focal loss parameters
        self.focal_alpha = self.config.get("focal_alpha", 0.25)
        self.focal_gamma = self.config.get("focal_gamma", 2.0)

        # Dice loss parameters
        self.dice_smooth = self.config.get("dice_smooth", 1e-6)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss

        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels

        Returns:
            Computed loss
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = self._apply_label_smoothing(targets)

        # Compute loss based on type
        if self.loss_type == "bce_with_logits":
            return self._bce_with_logits_loss(predictions, targets)

        elif self.loss_type == "bce":
            return self._bce_loss(predictions, targets)

        elif self.loss_type == "focal":
            return self._focal_loss(predictions, targets)

        elif self.loss_type == "dice":
            return self._dice_loss(predictions, targets)

        elif self.loss_type == "combined":
            return self._combined_loss(predictions, targets)

        elif self.loss_type == "weighted_bce":
            return self._weighted_bce_loss(predictions, targets)

        elif self.loss_type == "asymmetric":
            return self._asymmetric_loss(predictions, targets)

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def _apply_label_smoothing(self, targets: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to targets"""
        smoothing = self.label_smoothing
        targets = targets * (1 - smoothing) + 0.5 * smoothing
        return targets

    def _bce_with_logits_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Binary cross-entropy with logits loss"""
        return self.bce_loss(predictions, targets)

    def _bce_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Binary cross-entropy loss (expects probabilities)"""
        # Apply sigmoid if predictions are logits
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        return self.bce_loss_no_logits(predictions, targets)

    def _focal_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal loss for addressing class imbalance

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Focal loss
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)

        # Compute focal loss
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        # For positive samples
        pos_loss = -alpha * (1 - probs) ** gamma * targets * torch.log(probs + 1e-8)

        # For negative samples
        neg_loss = (
            -(1 - alpha) * probs**gamma * (1 - targets) * torch.log(1 - probs + 1e-8)
        )

        loss = pos_loss + neg_loss
        return loss.mean()

    def _dice_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Dice loss for binary segmentation-like problems

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Dice loss
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)

        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.dice_smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.dice_smooth
        )

        # Dice loss is 1 - Dice coefficient
        return 1.0 - dice_coeff

    def _combined_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Combined loss using multiple loss functions

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Combined loss
        """
        weights = self.config.get("combination_weights", {"bce": 0.7, "focal": 0.3})

        total_loss = 0.0

        if "bce" in weights:
            bce_loss = self._bce_with_logits_loss(predictions, targets)
            total_loss += weights["bce"] * bce_loss

        if "focal" in weights:
            focal_loss = self._focal_loss(predictions, targets)
            total_loss += weights["focal"] * focal_loss

        if "dice" in weights:
            dice_loss = self._dice_loss(predictions, targets)
            total_loss += weights["dice"] * dice_loss

        return total_loss

    def _weighted_bce_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Weighted BCE loss with dynamic class weights

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Weighted BCE loss
        """
        # Calculate class frequencies in current batch
        pos_count = targets.sum()
        neg_count = targets.numel() - pos_count

        # Avoid division by zero
        if pos_count == 0:
            pos_weight = 1.0
        elif neg_count == 0:
            pos_weight = 1.0
        else:
            pos_weight = neg_count / pos_count

        # Create weighted BCE loss
        weighted_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(predictions.device)
        )

        return weighted_bce(predictions, targets)

    def _asymmetric_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Asymmetric loss that penalizes false negatives more than false positives

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Asymmetric loss
        """
        probs = torch.sigmoid(predictions)

        # Asymmetric parameters
        gamma_pos = 1.0  # Focusing parameter for positive samples
        gamma_neg = 4.0  # Focusing parameter for negative samples

        # Positive samples (real videos misclassified as fake)
        pos_loss = -targets * (1 - probs) ** gamma_pos * torch.log(probs + 1e-8)

        # Negative samples (fake videos misclassified as real)
        neg_loss = -(1 - targets) * probs**gamma_neg * torch.log(1 - probs + 1e-8)

        loss = pos_loss + neg_loss
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative features
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize ContrastiveLoss

        Args:
            margin: Margin for contrastive loss
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self, features1: torch.Tensor, features2: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            features1: Features from first sample
            features2: Features from second sample
            labels: 1 if same class, 0 if different class

        Returns:
            Contrastive loss
        """
        # Compute Euclidean distance
        distance = F.pairwise_distance(features1, features2)

        # Contrastive loss
        loss = labels * distance.pow(2) + (1 - labels) * F.relu(
            self.margin - distance
        ).pow(2)

        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize TripletLoss

        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss

        Args:
            anchor: Anchor features
            positive: Positive features (same class as anchor)
            negative: Negative features (different class from anchor)

        Returns:
            Triplet loss
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)

        # Triplet loss
        loss = F.relu(pos_distance - neg_distance + self.margin)

        return loss.mean()


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss
    """

    def __init__(self, alpha: float = 0.7, temperature: float = 4.0):
        """
        Initialize DistillationLoss

        Args:
            alpha: Weight for distillation loss
            temperature: Temperature for softmax
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss

        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            labels: Ground truth labels

        Returns:
            Distillation loss
        """
        # Soft targets from teacher
        soft_targets = torch.sigmoid(teacher_logits / self.temperature)
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=1)

        # Distillation loss
        distill_loss = self.kl_div(soft_student, soft_targets) * (self.temperature**2)

        # Student loss
        student_loss = self.ce_loss(student_logits, labels)

        return self.alpha * distill_loss + (1 - self.alpha) * student_loss


# Factory function
def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss function from configuration

    Args:
        config: Loss configuration

    Returns:
        Loss function
    """
    loss_type = config.get("type", "bce_with_logits")

    if loss_type in [
        "bce_with_logits",
        "bce",
        "focal",
        "dice",
        "combined",
        "weighted_bce",
        "asymmetric",
    ]:
        return DeepfakeLoss(config)
    elif loss_type == "contrastive":
        return ContrastiveLoss(margin=config.get("margin", 1.0))
    elif loss_type == "triplet":
        return TripletLoss(margin=config.get("margin", 1.0))
    elif loss_type == "distillation":
        return DistillationLoss(
            alpha=config.get("alpha", 0.7), temperature=config.get("temperature", 4.0)
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


# Utility functions
def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets

    Args:
        labels: Array of labels

    Returns:
        Class weights tensor
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)

    weights = total_samples / (len(unique_labels) * counts)
    weight_dict = dict(zip(unique_labels, weights))

    # For binary classification, return positive class weight
    if len(unique_labels) == 2:
        return torch.tensor([weight_dict.get(1, 1.0)])
    else:
        return torch.tensor(
            [weight_dict.get(i, 1.0) for i in range(len(unique_labels))]
        )


def adaptive_loss_weighting(
    losses: Dict[str, torch.Tensor], epoch: int, total_epochs: int
) -> torch.Tensor:
    """
    Adaptive loss weighting that changes during training

    Args:
        losses: Dictionary of loss components
        epoch: Current epoch
        total_epochs: Total number of epochs

    Returns:
        Weighted combined loss
    """
    # Example: Start with more focus on basic BCE, gradually add focal loss
    progress = epoch / total_epochs

    bce_weight = 1.0 - 0.3 * progress
    focal_weight = 0.3 * progress

    total_loss = 0.0
    if "bce" in losses:
        total_loss += bce_weight * losses["bce"]
    if "focal" in losses:
        total_loss += focal_weight * losses["focal"]

    return total_loss

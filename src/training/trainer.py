"""
Enhanced Model Trainer for Deepfake Detection
Comprehensive training pipeline with advanced features
From Hasif's Workspace
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
from tqdm import tqdm

from .losses import DeepfakeLoss
from .metrics import DeepfakeMetrics
from ..models.model_utils import ModelUtils

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Enhanced trainer for deepfake detection models
    Supports advanced training techniques and monitoring
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Initialize ModelTrainer

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_dir = Path(checkpoint_dir)

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

        # Initialize components
        self._init_optimizer()
        self._init_scheduler()
        self._init_loss_function()
        self._init_metrics()

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rates": [],
        }

        # Mixed precision training
        self.use_amp = config.get("mixed_precision", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        logger.info(f"ModelTrainer initialized on device: {self.device}")
        logger.info(f"Mixed precision training: {self.use_amp}")

    def _init_optimizer(self):
        """Initialize optimizer"""
        optimizer_config = self.config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adam").lower()
        lr = optimizer_config.get("learning_rate", 0.001)
        weight_decay = optimizer_config.get("weight_decay", 0.0001)

        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get("betas", (0.9, 0.999)),
            )
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get("betas", (0.9, 0.999)),
            )
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=optimizer_config.get("momentum", 0.9),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        logger.info(f"Initialized {optimizer_type} optimizer with lr={lr}")

    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        scheduler_config = self.config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine").lower()

        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("num_epochs", 50),
                eta_min=scheduler_config.get("min_lr", 1e-6),
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1),
            )
        elif scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 5),
                verbose=True,
            )
        else:
            self.scheduler = None

        logger.info(f"Initialized {scheduler_type} scheduler")

    def _init_loss_function(self):
        """Initialize loss function"""
        loss_config = self.config.get("loss", {})
        self.criterion = DeepfakeLoss(loss_config)
        logger.info(f"Initialized loss function: {self.criterion}")

    def _init_metrics(self):
        """Initialize metrics"""
        metrics_config = self.config.get(
            "metrics", ["accuracy", "precision", "recall", "f1_score"]
        )
        self.metrics = DeepfakeMetrics(metrics_config)
        logger.info(f"Initialized metrics: {metrics_config}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # Handle video data (multiple frames per sample)
            if data.dim() == 5:  # (batch, frames, channels, height, width)
                batch_size, num_frames = data.shape[:2]
                data = data.view(-1, *data.shape[2:])  # Flatten frames
                target = target.repeat_interleave(num_frames)

            self.optimizer.zero_grad()

            if self.use_amp:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target.float().unsqueeze(1))

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(data)
                loss = self.criterion(output, target.float().unsqueeze(1))
                loss.backward()
                self.optimizer.step()

            # Collect predictions and labels
            predictions = torch.sigmoid(output).cpu().numpy()
            labels = target.cpu().numpy()

            all_predictions.extend(predictions.flatten())
            all_labels.extend(labels.flatten())

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics.calculate_metrics(
            np.array(all_labels), np.array(all_predictions)
        )

        return {"loss": avg_loss, **metrics}

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {self.current_epoch + 1} [Val]",
                leave=False,
            )

            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)

                # Handle video data
                if data.dim() == 5:
                    batch_size, num_frames = data.shape[:2]
                    data = data.view(-1, *data.shape[2:])
                    target = target.repeat_interleave(num_frames)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target.float().unsqueeze(1))
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target.float().unsqueeze(1))

                # Collect predictions and labels
                predictions = torch.sigmoid(output).cpu().numpy()
                labels = target.cpu().numpy()

                all_predictions.extend(predictions.flatten())
                all_labels.extend(labels.flatten())

                total_loss += loss.item()

                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Avg Loss": f"{total_loss / (len(progress_bar.n) + 1):.4f}",
                    }
                )

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics.calculate_metrics(
            np.array(all_labels), np.array(all_predictions)
        )

        return {"loss": avg_loss, **metrics}

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List]:
        """
        Train the model

        Args:
            num_epochs: Number of epochs to train (overrides config)

        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config.get("num_epochs", 50)

        logger.info(f"Starting training for {num_epochs} epochs")

        # Early stopping configuration
        early_stopping_patience = self.config.get("early_stopping_patience", 10)
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_results = self.train_epoch()

            # Validate epoch
            val_results = self.validate_epoch()

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results["accuracy"])
                else:
                    self.scheduler.step()

            # Log results
            current_lr = self.optimizer.param_groups[0]["lr"]
            self._log_epoch_results(train_results, val_results, current_lr)

            # Save checkpoint
            is_best = val_results["accuracy"] > self.best_metric
            if is_best:
                self.best_metric = val_results["accuracy"]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            self._save_checkpoint(val_results, is_best)

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.info("Training completed")
        return self.training_history

    def _log_epoch_results(
        self,
        train_results: Dict[str, float],
        val_results: Dict[str, float],
        learning_rate: float,
    ):
        """Log epoch results"""
        # Update history
        self.training_history["train_loss"].append(train_results["loss"])
        self.training_history["val_loss"].append(val_results["loss"])
        self.training_history["train_metrics"].append(train_results)
        self.training_history["val_metrics"].append(val_results)
        self.training_history["learning_rates"].append(learning_rate)

        # Log to console
        logger.info(f"Epoch {self.current_epoch + 1}:")
        logger.info(
            f"  Train - Loss: {train_results['loss']:.4f}, "
            f"Acc: {train_results['accuracy']:.4f}, "
            f"F1: {train_results.get('f1_score', 0):.4f}"
        )
        logger.info(
            f"  Val   - Loss: {val_results['loss']:.4f}, "
            f"Acc: {val_results['accuracy']:.4f}, "
            f"F1: {val_results.get('f1_score', 0):.4f}"
        )
        logger.info(f"  LR: {learning_rate:.6f}")

    def _save_checkpoint(self, val_results: Dict[str, float], is_best: bool):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": self.current_epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_metric": self.best_metric,
            "val_results": val_results,
            "training_history": self.training_history,
            "config": self.config,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)

            # Also save just the model state dict
            model_path = self.checkpoint_dir / "deepfake_detector_best.pth"
            torch.save(self.model.state_dict(), model_path)

            logger.info(
                f"New best model saved with accuracy: {val_results['accuracy']:.4f}"
            )

        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", 0.0)
        self.training_history = checkpoint.get(
            "training_history",
            {
                "train_loss": [],
                "val_loss": [],
                "train_metrics": [],
                "val_metrics": [],
                "learning_rates": [],
            },
        )

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def get_model_summary(self) -> str:
        """Get model summary"""
        return ModelUtils.model_summary(self.model, (3, 224, 224))

    def export_model(self, export_path: str, format: str = "onnx"):
        """Export trained model"""
        if format.lower() == "onnx":
            ModelUtils.convert_to_onnx(
                self.model, export_path, input_size=(3, 224, 224)
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Factory function
def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    **kwargs,
) -> ModelTrainer:
    """Factory function to create ModelTrainer"""
    return ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        **kwargs,
    )

"""
Model utilities for deepfake detection
Helper functions for model operations, loading, saving, and optimization
From Hasif's Workspace
"""

import torch
import torch.nn as nn
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ModelUtils:
    """Utility class for model operations"""

    @staticmethod
    def save_model(
        model: nn.Module,
        filepath: str,
        metadata: Optional[Dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
    ):
        """
        Save model with metadata

        Args:
            model: PyTorch model to save
            filepath: Path to save the model
            metadata: Additional metadata to save
            optimizer: Optimizer state to save
            epoch: Current epoch
            loss: Current loss value
        """
        try:
            # Prepare save dictionary
            save_dict = {
                "model_state_dict": model.state_dict(),
                "model_info": getattr(model, "model_info", {}),
                "timestamp": torch.tensor(torch.now()).item()
                if hasattr(torch, "now")
                else None,
            }

            # Add optional components
            if optimizer is not None:
                save_dict["optimizer_state_dict"] = optimizer.state_dict()

            if epoch is not None:
                save_dict["epoch"] = epoch

            if loss is not None:
                save_dict["loss"] = loss

            if metadata is not None:
                save_dict["metadata"] = metadata

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save model
            torch.save(save_dict, filepath)

            # Save metadata as JSON for easy reading
            metadata_path = filepath.replace(".pth", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "model_info": save_dict.get("model_info", {}),
                        "metadata": save_dict.get("metadata", {}),
                        "epoch": save_dict.get("epoch"),
                        "loss": save_dict.get("loss"),
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Model saved successfully to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @staticmethod
    def load_model(
        model: nn.Module,
        filepath: str,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load model from file

        Args:
            model: Model instance to load weights into
            filepath: Path to model file
            device: Device to load model on
            strict: Whether to strictly enforce state dict keys

        Returns:
            Dictionary with loaded information
        """
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load checkpoint
            checkpoint = torch.load(filepath, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

            # Move model to device
            model.to(device)

            logger.info(f"Model loaded successfully from {filepath}")

            return {
                "model_info": checkpoint.get("model_info", {}),
                "metadata": checkpoint.get("metadata", {}),
                "epoch": checkpoint.get("epoch"),
                "loss": checkpoint.get("loss"),
                "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
            }

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count model parameters

        Args:
            model: PyTorch model

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "total_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }

    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """
        Get model size in MB

        Args:
            model: PyTorch model

        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb

    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: list):
        """
        Freeze specific layers

        Args:
            model: PyTorch model
            layer_names: List of layer names to freeze
        """
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")

    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: list):
        """
        Unfreeze specific layers

        Args:
            model: PyTorch model
            layer_names: List of layer names to unfreeze
        """
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")

    @staticmethod
    def get_layer_info(model: nn.Module) -> Dict[str, Dict]:
        """
        Get information about model layers

        Args:
            model: PyTorch model

        Returns:
            Dictionary with layer information
        """
        layer_info = {}

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info[name] = {
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "trainable": any(p.requires_grad for p in module.parameters()),
                }

        return layer_info

    @staticmethod
    def model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
        """
        Generate model summary

        Args:
            model: PyTorch model
            input_size: Input tensor size (C, H, W)

        Returns:
            Model summary string
        """

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = f"{class_name}-{module_idx + 1}"
                summary[m_key] = {
                    "input_shape": list(input[0].size()),
                    "output_shape": list(output.size()),
                    "nb_params": sum([p.numel() for p in module.parameters()]),
                }

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = next(model.parameters()).device
        model.eval()

        summary = {}
        hooks = []

        model.apply(register_hook)

        # Create dummy input
        x = torch.randn(1, *input_size).to(device)
        model(x)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Format summary
        summary_str = "Model Summary:\n"
        summary_str += "-" * 80 + "\n"
        summary_str += f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}\n"
        summary_str += "=" * 80 + "\n"

        total_params = 0
        for layer_name, layer_info in summary.items():
            summary_str += f"{layer_name:<25} {str(layer_info['output_shape']):<20} {layer_info['nb_params']:<15}\n"
            total_params += layer_info["nb_params"]

        summary_str += "=" * 80 + "\n"
        summary_str += f"Total params: {total_params:,}\n"
        summary_str += f"Model size: {ModelUtils.get_model_size(model):.2f} MB\n"

        return summary_str

    @staticmethod
    def convert_to_onnx(
        model: nn.Module,
        filepath: str,
        input_size: Tuple[int, ...] = (3, 224, 224),
        batch_size: int = 1,
    ):
        """
        Convert model to ONNX format

        Args:
            model: PyTorch model
            filepath: Output ONNX file path
            input_size: Input tensor size
            batch_size: Batch size for export
        """
        try:
            device = next(model.parameters()).device
            model.eval()

            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_size).to(device)

            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                filepath,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            logger.info(f"Model exported to ONNX: {filepath}")

        except Exception as e:
            logger.error(f"Error converting to ONNX: {e}")
            raise

    @staticmethod
    def quantize_model(model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
        """
        Quantize model for deployment

        Args:
            model: PyTorch model
            calibration_data: Data for calibration

        Returns:
            Quantized model
        """
        try:
            model.eval()

            # Prepare model for quantization
            model_prepared = torch.quantization.prepare(model)

            # Calibrate with sample data
            with torch.no_grad():
                model_prepared(calibration_data)

            # Convert to quantized model
            model_quantized = torch.quantization.convert(model_prepared)

            logger.info("Model quantized successfully")
            return model_quantized

        except Exception as e:
            logger.error(f"Error quantizing model: {e}")
            raise


# Utility functions
def load_pretrained_weights(model: nn.Module, weights_path: str, strict: bool = True):
    """Load pretrained weights into model"""
    return ModelUtils.load_model(model, weights_path, strict=strict)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
):
    """Save training checkpoint"""
    ModelUtils.save_model(model, filepath, optimizer=optimizer, epoch=epoch, loss=loss)


def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device()
        if torch.cuda.is_available()
        else None,
        "device_name": torch.cuda.get_device_name()
        if torch.cuda.is_available()
        else "CPU",
    }

    if torch.cuda.is_available():
        device_info["memory_allocated"] = torch.cuda.memory_allocated()
        device_info["memory_reserved"] = torch.cuda.memory_reserved()

    return device_info

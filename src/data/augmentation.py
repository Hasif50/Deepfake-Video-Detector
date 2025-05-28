"""
Advanced Data Augmentation for Deepfake Detection
Comprehensive augmentation strategies for video and image data
From Hasif's Workspace
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import random
from typing import List, Tuple, Optional, Dict, Any, Callable
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger(__name__)


class VideoAugmentation:
    """Advanced video augmentation for deepfake detection"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize VideoAugmentation

        Args:
            config: Configuration dictionary for augmentation parameters
        """
        self.config = config or self._get_default_config()

        # Initialize augmentation functions
        self.spatial_augs = self._init_spatial_augmentations()
        self.temporal_augs = self._init_temporal_augmentations()
        self.quality_augs = self._init_quality_augmentations()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default augmentation configuration"""
        return {
            "spatial": {
                "horizontal_flip": 0.5,
                "rotation_range": 10,
                "scale_range": [0.9, 1.1],
                "translation_range": 0.1,
                "shear_range": 5,
                "perspective_distortion": 0.1,
            },
            "temporal": {
                "frame_dropout": 0.1,
                "temporal_shift": 2,
                "speed_change": [0.8, 1.2],
                "reverse_probability": 0.1,
            },
            "quality": {
                "brightness_range": [0.8, 1.2],
                "contrast_range": [0.8, 1.2],
                "saturation_range": [0.8, 1.2],
                "hue_range": [-0.1, 0.1],
                "gamma_range": [0.8, 1.2],
                "gaussian_blur": 0.1,
                "noise_std": 0.01,
                "jpeg_compression": [70, 95],
            },
            "advanced": {
                "mixup_alpha": 0.2,
                "cutmix_alpha": 1.0,
                "cutout_ratio": 0.1,
                "grid_distortion": 0.1,
                "elastic_transform": 0.1,
            },
        }

    def _init_spatial_augmentations(self) -> List[Callable]:
        """Initialize spatial augmentation functions"""
        return [
            self._random_horizontal_flip,
            self._random_rotation,
            self._random_scale,
            self._random_translation,
            self._random_shear,
            self._random_perspective,
        ]

    def _init_temporal_augmentations(self) -> List[Callable]:
        """Initialize temporal augmentation functions"""
        return [
            self._random_frame_dropout,
            self._random_temporal_shift,
            self._random_speed_change,
            self._random_reverse,
        ]

    def _init_quality_augmentations(self) -> List[Callable]:
        """Initialize quality augmentation functions"""
        return [
            self._random_brightness,
            self._random_contrast,
            self._random_saturation,
            self._random_hue,
            self._random_gamma,
            self._random_gaussian_blur,
            self._random_noise,
            self._random_jpeg_compression,
        ]

    def augment_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply augmentations to video frames

        Args:
            frames: List of video frames as numpy arrays

        Returns:
            Augmented frames
        """
        if not frames:
            return frames

        # Apply temporal augmentations first
        frames = self._apply_temporal_augmentations(frames)

        # Apply spatial and quality augmentations to each frame
        augmented_frames = []
        for frame in frames:
            # Apply spatial augmentations
            frame = self._apply_spatial_augmentations(frame)

            # Apply quality augmentations
            frame = self._apply_quality_augmentations(frame)

            augmented_frames.append(frame)

        return augmented_frames

    def _apply_temporal_augmentations(
        self, frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Apply temporal augmentations to video sequence"""
        for aug_func in self.temporal_augs:
            if random.random() < 0.3:  # Apply each temporal aug with 30% probability
                frames = aug_func(frames)

        return frames

    def _apply_spatial_augmentations(self, frame: np.ndarray) -> np.ndarray:
        """Apply spatial augmentations to a single frame"""
        for aug_func in self.spatial_augs:
            if random.random() < 0.3:  # Apply each spatial aug with 30% probability
                frame = aug_func(frame)

        return frame

    def _apply_quality_augmentations(self, frame: np.ndarray) -> np.ndarray:
        """Apply quality augmentations to a single frame"""
        for aug_func in self.quality_augs:
            if random.random() < 0.3:  # Apply each quality aug with 30% probability
                frame = aug_func(frame)

        return frame

    # Spatial Augmentations
    def _random_horizontal_flip(self, frame: np.ndarray) -> np.ndarray:
        """Random horizontal flip"""
        if random.random() < self.config["spatial"]["horizontal_flip"]:
            return cv2.flip(frame, 1)
        return frame

    def _random_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Random rotation"""
        angle_range = self.config["spatial"]["rotation_range"]
        angle = random.uniform(-angle_range, angle_range)

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, rotation_matrix, (w, h))

    def _random_scale(self, frame: np.ndarray) -> np.ndarray:
        """Random scaling"""
        scale_range = self.config["spatial"]["scale_range"]
        scale = random.uniform(scale_range[0], scale_range[1])

        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize and crop/pad to original size
        resized = cv2.resize(frame, (new_w, new_h))

        if scale > 1.0:
            # Crop to original size
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h : start_h + h, start_w : start_w + w]
        else:
            # Pad to original size
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            return cv2.copyMakeBorder(
                resized,
                pad_h,
                h - new_h - pad_h,
                pad_w,
                w - new_w - pad_w,
                cv2.BORDER_REFLECT,
            )

    def _random_translation(self, frame: np.ndarray) -> np.ndarray:
        """Random translation"""
        h, w = frame.shape[:2]
        translation_range = self.config["spatial"]["translation_range"]

        tx = random.uniform(-translation_range, translation_range) * w
        ty = random.uniform(-translation_range, translation_range) * h

        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(frame, translation_matrix, (w, h))

    def _random_shear(self, frame: np.ndarray) -> np.ndarray:
        """Random shear transformation"""
        shear_range = self.config["spatial"]["shear_range"]
        shear_x = random.uniform(-shear_range, shear_range)
        shear_y = random.uniform(-shear_range, shear_range)

        h, w = frame.shape[:2]

        # Create shear transformation matrix
        shear_matrix = np.float32(
            [[1, np.tan(np.radians(shear_x)), 0], [np.tan(np.radians(shear_y)), 1, 0]]
        )

        return cv2.warpAffine(frame, shear_matrix, (w, h))

    def _random_perspective(self, frame: np.ndarray) -> np.ndarray:
        """Random perspective transformation"""
        distortion = self.config["spatial"]["perspective_distortion"]
        h, w = frame.shape[:2]

        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Add random distortion to destination points
        dst_points = src_points.copy()
        for i in range(4):
            dst_points[i][0] += random.uniform(-distortion * w, distortion * w)
            dst_points[i][1] += random.uniform(-distortion * h, distortion * h)

        # Apply perspective transformation
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(frame, perspective_matrix, (w, h))

    # Temporal Augmentations
    def _random_frame_dropout(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Random frame dropout"""
        dropout_rate = self.config["temporal"]["frame_dropout"]

        if len(frames) <= 2:  # Don't drop frames if too few
            return frames

        num_to_drop = int(len(frames) * dropout_rate)
        if num_to_drop == 0:
            return frames

        # Randomly select frames to drop
        indices_to_keep = random.sample(range(len(frames)), len(frames) - num_to_drop)
        indices_to_keep.sort()

        return [frames[i] for i in indices_to_keep]

    def _random_temporal_shift(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Random temporal shift (circular shift)"""
        max_shift = self.config["temporal"]["temporal_shift"]
        shift = random.randint(-max_shift, max_shift)

        if shift == 0 or len(frames) <= abs(shift):
            return frames

        return frames[shift:] + frames[:shift]

    def _random_speed_change(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Random speed change (frame sampling)"""
        speed_range = self.config["temporal"]["speed_change"]
        speed_factor = random.uniform(speed_range[0], speed_range[1])

        if speed_factor == 1.0:
            return frames

        # Resample frames based on speed factor
        original_indices = np.arange(len(frames))
        new_indices = np.linspace(0, len(frames) - 1, int(len(frames) / speed_factor))
        new_indices = np.clip(new_indices, 0, len(frames) - 1).astype(int)

        return [frames[i] for i in new_indices]

    def _random_reverse(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Random temporal reversal"""
        if random.random() < self.config["temporal"]["reverse_probability"]:
            return frames[::-1]
        return frames

    # Quality Augmentations
    def _random_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Random brightness adjustment"""
        brightness_range = self.config["quality"]["brightness_range"]
        factor = random.uniform(brightness_range[0], brightness_range[1])

        return np.clip(frame * factor, 0, 255).astype(np.uint8)

    def _random_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Random contrast adjustment"""
        contrast_range = self.config["quality"]["contrast_range"]
        factor = random.uniform(contrast_range[0], contrast_range[1])

        mean = np.mean(frame)
        return np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)

    def _random_saturation(self, frame: np.ndarray) -> np.ndarray:
        """Random saturation adjustment"""
        saturation_range = self.config["quality"]["saturation_range"]
        factor = random.uniform(saturation_range[0], saturation_range[1])

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _random_hue(self, frame: np.ndarray) -> np.ndarray:
        """Random hue adjustment"""
        hue_range = self.config["quality"]["hue_range"]
        shift = random.uniform(hue_range[0], hue_range[1]) * 180

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _random_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Random gamma correction"""
        gamma_range = self.config["quality"]["gamma_range"]
        gamma = random.uniform(gamma_range[0], gamma_range[1])

        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            np.uint8
        )

        return cv2.LUT(frame, table)

    def _random_gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        """Random Gaussian blur"""
        if random.random() < self.config["quality"]["gaussian_blur"]:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
        return frame

    def _random_noise(self, frame: np.ndarray) -> np.ndarray:
        """Random Gaussian noise"""
        noise_std = self.config["quality"]["noise_std"]
        noise = np.random.normal(0, noise_std * 255, frame.shape).astype(np.float32)

        noisy_frame = frame.astype(np.float32) + noise
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)

    def _random_jpeg_compression(self, frame: np.ndarray) -> np.ndarray:
        """Random JPEG compression"""
        quality_range = self.config["quality"]["jpeg_compression"]
        quality = random.randint(quality_range[0], quality_range[1])

        # Encode and decode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode_param
        )
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

        return cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)


# PyTorch Transform Classes
class RandomVideoAugmentation:
    """PyTorch-compatible video augmentation transform"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.augmenter = VideoAugmentation(config)

    def __call__(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply augmentation to tensor frames"""
        # Convert tensors to numpy arrays
        numpy_frames = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                # Convert from CHW to HWC and to numpy
                frame_np = frame.permute(1, 2, 0).numpy()
                frame_np = (frame_np * 255).astype(np.uint8)
                numpy_frames.append(frame_np)
            else:
                numpy_frames.append(frame)

        # Apply augmentation
        augmented_frames = self.augmenter.augment_video(numpy_frames)

        # Convert back to tensors
        tensor_frames = []
        for frame in augmented_frames:
            frame_tensor = torch.from_numpy(frame).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
            tensor_frames.append(frame_tensor)

        return tensor_frames


# Factory function
def create_augmentation_pipeline(
    config: Optional[Dict[str, Any]] = None,
) -> VideoAugmentation:
    """Create augmentation pipeline from configuration"""
    return VideoAugmentation(config)

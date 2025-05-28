"""
Enhanced Dataset Classes for Deepfake Detection
PyTorch dataset implementations with advanced features
From Hasif's Workspace
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import json
import logging
from typing import List, Tuple, Optional, Dict, Any, Callable
from pathlib import Path
import pandas as pd
from torchvision import transforms
import random

logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """
    Enhanced PyTorch Dataset for deepfake detection
    Supports various data formats and advanced preprocessing
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_frames_per_video: int = 50,
        frame_sampling_strategy: str = "uniform",
        cache_frames: bool = False,
        metadata_file: Optional[str] = None,
    ):
        """
        Initialize DeepfakeDataset

        Args:
            data_dir: Root directory containing the dataset
            split: Dataset split (train, val, test)
            transform: Transform to apply to frames
            target_transform: Transform to apply to labels
            max_frames_per_video: Maximum frames to load per video
            frame_sampling_strategy: Strategy for frame sampling
            cache_frames: Whether to cache frames in memory
            metadata_file: Optional metadata file with labels
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.max_frames_per_video = max_frames_per_video
        self.frame_sampling_strategy = frame_sampling_strategy
        self.cache_frames = cache_frames

        # Initialize data structures
        self.samples = []
        self.labels = []
        self.metadata = {}
        self.frame_cache = {} if cache_frames else None

        # Load dataset
        self._load_dataset(metadata_file)

        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_dataset(self, metadata_file: Optional[str]):
        """Load dataset samples and labels"""
        if metadata_file and os.path.exists(metadata_file):
            self._load_from_metadata(metadata_file)
        else:
            self._load_from_directory_structure()

    def _load_from_metadata(self, metadata_file: str):
        """Load dataset from metadata file (CSV or JSON)"""
        try:
            if metadata_file.endswith(".csv"):
                df = pd.read_csv(metadata_file)
                for _, row in df.iterrows():
                    if row.get("split", "train") == self.split:
                        self.samples.append(row["path"])
                        self.labels.append(int(row["label"]))
                        if "metadata" in row:
                            self.metadata[row["path"]] = json.loads(row["metadata"])

            elif metadata_file.endswith(".json"):
                with open(metadata_file, "r") as f:
                    data = json.load(f)

                for item in data:
                    if item.get("split", "train") == self.split:
                        self.samples.append(item["path"])
                        self.labels.append(int(item["label"]))
                        self.metadata[item["path"]] = item.get("metadata", {})

        except Exception as e:
            logger.error(f"Error loading metadata file {metadata_file}: {e}")
            self._load_from_directory_structure()

    def _load_from_directory_structure(self):
        """Load dataset from directory structure (real/fake folders)"""
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            logger.warning(
                f"Split directory {split_dir} not found, using root directory"
            )
            split_dir = self.data_dir

        # Look for real and fake subdirectories
        real_dir = split_dir / "real"
        fake_dir = split_dir / "fake"

        # Load real samples (label = 0)
        if real_dir.exists():
            for file_path in real_dir.glob("*"):
                if self._is_valid_sample(file_path):
                    self.samples.append(str(file_path))
                    self.labels.append(0)

        # Load fake samples (label = 1)
        if fake_dir.exists():
            for file_path in fake_dir.glob("*"):
                if self._is_valid_sample(file_path):
                    self.samples.append(str(file_path))
                    self.labels.append(1)

        # If no subdirectories, try to infer from filenames
        if not self.samples:
            self._load_from_filenames(split_dir)

    def _load_from_filenames(self, directory: Path):
        """Load samples by inferring labels from filenames"""
        for file_path in directory.glob("*"):
            if self._is_valid_sample(file_path):
                filename = file_path.name.lower()

                # Infer label from filename
                if any(
                    keyword in filename for keyword in ["fake", "deepfake", "synthetic"]
                ):
                    label = 1
                elif any(
                    keyword in filename for keyword in ["real", "authentic", "original"]
                ):
                    label = 0
                else:
                    # Default to real if can't determine
                    label = 0

                self.samples.append(str(file_path))
                self.labels.append(label)

    def _is_valid_sample(self, file_path: Path) -> bool:
        """Check if file is a valid sample"""
        if file_path.is_dir():
            # Check if directory contains frames
            frame_files = list(file_path.glob("*.jpg")) + list(file_path.glob("*.png"))
            return len(frame_files) > 0
        else:
            # Check if it's a video file
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
            return file_path.suffix.lower() in video_extensions

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset"""
        sample_path = self.samples[idx]
        label = self.labels[idx]

        # Load frames
        frames = self._load_frames(sample_path)

        # Apply transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames into tensor
        if frames:
            # If multiple frames, stack them
            if len(frames) > 1:
                frame_tensor = torch.stack(frames)
            else:
                frame_tensor = frames[0]
        else:
            # Return dummy tensor if no frames
            frame_tensor = torch.zeros(3, 224, 224)

        # Apply target transform
        if self.target_transform:
            label = self.target_transform(label)

        return frame_tensor, label

    def _load_frames(self, sample_path: str) -> List[np.ndarray]:
        """Load frames from sample path"""
        # Check cache first
        if self.frame_cache and sample_path in self.frame_cache:
            return self.frame_cache[sample_path]

        frames = []
        sample_path = Path(sample_path)

        if sample_path.is_dir():
            # Load from frame directory
            frames = self._load_frames_from_directory(sample_path)
        else:
            # Load from video file
            frames = self._load_frames_from_video(sample_path)

        # Cache frames if enabled
        if self.frame_cache:
            self.frame_cache[sample_path] = frames

        return frames

    def _load_frames_from_directory(self, frame_dir: Path) -> List[np.ndarray]:
        """Load frames from directory of images"""
        frame_files = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))

        # Sample frames based on strategy
        if len(frame_files) > self.max_frames_per_video:
            frame_files = self._sample_frames(frame_files)

        frames = []
        for frame_file in frame_files:
            try:
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            except Exception as e:
                logger.warning(f"Error loading frame {frame_file}: {e}")

        return frames

    def _load_frames_from_video(self, video_path: Path) -> List[np.ndarray]:
        """Load frames from video file"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate frame indices to extract
            if total_frames > self.max_frames_per_video:
                frame_indices = self._get_frame_indices(total_frames)
            else:
                frame_indices = list(range(total_frames))

            frames = []
            current_frame = 0

            while cap.isOpened() and len(frames) < len(frame_indices):
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame in frame_indices:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                current_frame += 1

            cap.release()
            return frames

        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return []

    def _sample_frames(self, frame_files: List[Path]) -> List[Path]:
        """Sample frames based on strategy"""
        if self.frame_sampling_strategy == "uniform":
            # Uniform sampling
            indices = np.linspace(
                0, len(frame_files) - 1, self.max_frames_per_video, dtype=int
            )
            return [frame_files[i] for i in indices]

        elif self.frame_sampling_strategy == "random":
            # Random sampling
            return random.sample(frame_files, self.max_frames_per_video)

        elif self.frame_sampling_strategy == "first":
            # Take first N frames
            return frame_files[: self.max_frames_per_video]

        elif self.frame_sampling_strategy == "last":
            # Take last N frames
            return frame_files[-self.max_frames_per_video :]

        else:
            # Default to uniform
            indices = np.linspace(
                0, len(frame_files) - 1, self.max_frames_per_video, dtype=int
            )
            return [frame_files[i] for i in indices]

    def _get_frame_indices(self, total_frames: int) -> List[int]:
        """Get frame indices based on sampling strategy"""
        if self.frame_sampling_strategy == "uniform":
            return np.linspace(
                0, total_frames - 1, self.max_frames_per_video, dtype=int
            ).tolist()

        elif self.frame_sampling_strategy == "random":
            return sorted(random.sample(range(total_frames), self.max_frames_per_video))

        elif self.frame_sampling_strategy == "first":
            return list(range(min(self.max_frames_per_video, total_frames)))

        elif self.frame_sampling_strategy == "last":
            start_idx = max(0, total_frames - self.max_frames_per_video)
            return list(range(start_idx, total_frames))

        else:
            return np.linspace(
                0, total_frames - 1, self.max_frames_per_video, dtype=int
            ).tolist()

    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution in the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample"""
        sample_path = self.samples[idx]
        return self.metadata.get(sample_path, {})


class FrameDataset(Dataset):
    """Dataset for individual frames (not videos)"""

    def __init__(
        self,
        frame_dir: str,
        labels_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize FrameDataset

        Args:
            frame_dir: Directory containing frame images
            labels_file: CSV file with frame labels
            transform: Transform to apply to frames
            target_transform: Transform to apply to labels
        """
        self.frame_dir = Path(frame_dir)
        self.transform = transform
        self.target_transform = target_transform

        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        self.samples = self.labels_df["filename"].tolist()
        self.labels = self.labels_df["label"].tolist()

        logger.info(f"Loaded {len(self.samples)} frame samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a frame sample"""
        filename = self.samples[idx]
        label = self.labels[idx]

        # Load frame
        frame_path = self.frame_dir / filename
        frame = cv2.imread(str(frame_path))

        if frame is None:
            # Return dummy frame if loading fails
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            frame = self.transform(frame)

        if self.target_transform:
            label = self.target_transform(label)

        return frame, label


# Utility functions for creating data loaders
def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform_config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders

    Args:
        data_dir: Root data directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        transform_config: Configuration for data transforms

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Default transforms
    if transform_config is None:
        transform_config = {
            "resize": (224, 224),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        }

    # Create transforms
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(transform_config["resize"]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_config["normalize_mean"],
                std=transform_config["normalize_std"],
            ),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(transform_config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_config["normalize_mean"],
                std=transform_config["normalize_std"],
            ),
        ]
    )

    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=data_dir, split="train", transform=train_transform
    )

    val_dataset = DeepfakeDataset(
        data_dir=data_dir, split="val", transform=val_test_transform
    )

    test_dataset = DeepfakeDataset(
        data_dir=data_dir, split="test", transform=val_test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def collate_video_frames(batch):
    """Custom collate function for variable-length video sequences"""
    frames, labels = zip(*batch)

    # Handle variable number of frames per video
    max_frames = max(f.shape[0] if f.dim() > 3 else 1 for f in frames)

    # Pad sequences to same length
    padded_frames = []
    for frame_seq in frames:
        if frame_seq.dim() == 3:
            # Single frame, add frame dimension
            frame_seq = frame_seq.unsqueeze(0)

        # Pad to max_frames
        if frame_seq.shape[0] < max_frames:
            padding = torch.zeros(max_frames - frame_seq.shape[0], *frame_seq.shape[1:])
            frame_seq = torch.cat([frame_seq, padding], dim=0)

        padded_frames.append(frame_seq)

    return torch.stack(padded_frames), torch.tensor(labels)

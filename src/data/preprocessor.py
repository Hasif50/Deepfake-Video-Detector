"""
Enhanced Video Preprocessor
Advanced video preprocessing with optimizations and features
From Hasif's Workspace
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import concurrent.futures
import multiprocessing

logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """Enhanced video preprocessing with advanced features"""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        frame_rate: int = 5,
        max_frames: int = 50,
        quality_threshold: float = 0.1,
        num_workers: Optional[int] = None,
    ):
        """
        Initialize VideoPreprocessor

        Args:
            target_size: Target frame size (width, height)
            frame_rate: Frames per second to extract
            max_frames: Maximum frames to extract per video
            quality_threshold: Minimum quality threshold for frames
            num_workers: Number of worker processes for parallel processing
        """
        self.target_size = target_size
        self.frame_rate = frame_rate
        self.max_frames = max_frames
        self.quality_threshold = quality_threshold
        self.num_workers = num_workers or multiprocessing.cpu_count()

        # Supported video formats
        self.supported_formats = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
        }

        logger.info(f"VideoPreprocessor initialized with {self.num_workers} workers")

    def extract_frames_from_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        save_frames: bool = False,
    ) -> List[np.ndarray]:
        """
        Extract frames from a single video

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames (if save_frames=True)
            save_frames: Whether to save frames to disk

        Returns:
            List of extracted frames as numpy arrays
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Validate file format
            file_ext = Path(video_path).suffix.lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported video format: {file_ext}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0 or video_fps == 0:
                raise ValueError("Invalid video metadata")

            # Calculate frame indices to extract
            frame_indices = self._calculate_frame_indices(total_frames, video_fps)

            # Extract frames
            frames = []
            current_frame = 0

            while cap.isOpened() and len(frames) < len(frame_indices):
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame in frame_indices:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Quality check
                    if self._check_frame_quality(frame_rgb):
                        # Resize frame
                        frame_resized = cv2.resize(
                            frame_rgb, self.target_size, interpolation=cv2.INTER_LINEAR
                        )

                        frames.append(frame_resized)

                        # Save frame if requested
                        if save_frames and output_dir:
                            self._save_frame(frame_resized, output_dir, len(frames))

                current_frame += 1

            cap.release()

            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            if "cap" in locals():
                cap.release()
            return []

    def _calculate_frame_indices(
        self, total_frames: int, video_fps: float
    ) -> List[int]:
        """Calculate which frame indices to extract"""
        # Calculate target number of frames
        target_frames = min(
            self.max_frames, max(1, int(total_frames * self.frame_rate / video_fps))
        )

        if target_frames >= total_frames:
            return list(range(total_frames))

        # Evenly distribute frames across video
        return np.linspace(0, total_frames - 1, target_frames, dtype=int).tolist()

    def _check_frame_quality(self, frame: np.ndarray) -> bool:
        """
        Check if frame meets quality threshold

        Args:
            frame: Frame to check

        Returns:
            True if frame quality is acceptable
        """
        try:
            # Convert to grayscale for quality assessment
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Calculate Laplacian variance (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Normalize by image size
            normalized_var = laplacian_var / (gray.shape[0] * gray.shape[1])

            return normalized_var > self.quality_threshold

        except Exception as e:
            logger.warning(f"Error checking frame quality: {e}")
            return True  # Accept frame if quality check fails

    def _save_frame(self, frame: np.ndarray, output_dir: str, frame_number: int):
        """Save frame to disk"""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            filename = f"frame_{frame_number:06d}.jpg"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, frame_bgr)

        except Exception as e:
            logger.warning(f"Error saving frame {frame_number}: {e}")

    def process_video_batch(
        self,
        video_paths: List[str],
        output_base_dir: Optional[str] = None,
        save_frames: bool = False,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Process multiple videos in parallel

        Args:
            video_paths: List of video file paths
            output_base_dir: Base directory for saving frames
            save_frames: Whether to save frames to disk

        Returns:
            Dictionary mapping video paths to extracted frames
        """
        logger.info(
            f"Processing {len(video_paths)} videos with {self.num_workers} workers"
        )

        results = {}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            # Submit tasks
            future_to_path = {}

            for video_path in video_paths:
                output_dir = None
                if save_frames and output_base_dir:
                    video_name = Path(video_path).stem
                    output_dir = os.path.join(output_base_dir, video_name)

                future = executor.submit(
                    self.extract_frames_from_video, video_path, output_dir, save_frames
                )
                future_to_path[future] = video_path

            # Collect results
            for future in concurrent.futures.as_completed(future_to_path):
                video_path = future_to_path[future]
                try:
                    frames = future.result()
                    results[video_path] = frames
                except Exception as e:
                    logger.error(f"Error processing {video_path}: {e}")
                    results[video_path] = []

        logger.info(f"Completed processing {len(video_paths)} videos")
        return results

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get comprehensive video information

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}

            info = {
                "path": video_path,
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": 0,
                "file_size": os.path.getsize(video_path),
                "format": Path(video_path).suffix.lower(),
            }

            # Calculate duration
            if info["fps"] > 0:
                info["duration"] = info["total_frames"] / info["fps"]

            cap.release()
            return info

        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
            return {}

    def validate_video(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate video file

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file existence
            if not os.path.exists(video_path):
                return False, "File does not exist"

            # Check file extension
            file_ext = Path(video_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return False, f"Unsupported format: {file_ext}"

            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                return False, "File is empty"

            # Try to open with OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file"

            # Check basic properties
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            cap.release()

            if frame_count <= 0:
                return False, "Video has no frames"

            if fps <= 0:
                return False, "Invalid frame rate"

            return True, "Valid video file"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def set_quality_threshold(self, threshold: float):
        """Set quality threshold for frame filtering"""
        self.quality_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Quality threshold set to {self.quality_threshold}")

    def get_supported_formats(self) -> set:
        """Get supported video formats"""
        return self.supported_formats.copy()


# Utility functions
def create_video_preprocessor(config: Dict[str, Any]) -> VideoPreprocessor:
    """Factory function to create VideoPreprocessor from config"""
    return VideoPreprocessor(
        target_size=tuple(config.get("target_size", [224, 224])),
        frame_rate=config.get("frame_rate", 5),
        max_frames=config.get("max_frames", 50),
        quality_threshold=config.get("quality_threshold", 0.1),
        num_workers=config.get("num_workers"),
    )


def batch_process_videos(
    video_directory: str,
    output_directory: str,
    preprocessor: Optional[VideoPreprocessor] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Batch process all videos in a directory

    Args:
        video_directory: Directory containing videos
        output_directory: Directory to save processed frames
        preprocessor: VideoPreprocessor instance (creates default if None)

    Returns:
        Dictionary mapping video paths to extracted frames
    """
    if preprocessor is None:
        preprocessor = VideoPreprocessor()

    # Find all video files
    video_paths = []
    for ext in preprocessor.get_supported_formats():
        pattern = f"*{ext}"
        video_paths.extend(Path(video_directory).glob(pattern))

    video_paths = [str(path) for path in video_paths]

    if not video_paths:
        logger.warning(f"No video files found in {video_directory}")
        return {}

    # Process videos
    return preprocessor.process_video_batch(
        video_paths, output_directory, save_frames=True
    )

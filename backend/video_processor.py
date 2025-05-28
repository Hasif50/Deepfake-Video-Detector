"""
Video Processing Module
Handles video file operations, frame extraction, and video analysis
From Hasif's Workspace
"""

import cv2
import numpy as np
import os
import tempfile
import asyncio
import logging
from typing import List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing operations for deepfake detection"""

    def __init__(self):
        self.supported_formats = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
        }
        self.max_file_size = 500 * 1024 * 1024  # 500MB

    async def extract_frames(
        self, video_path: str, num_frames: int = 5, frame_rate: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video file

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            frame_rate: Target frame rate for extraction (None for auto)

        Returns:
            List of frames as numpy arrays (RGB format)
        """
        try:
            # Validate video file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size > self.max_file_size:
                raise ValueError(
                    f"File too large: {file_size} bytes (max: {self.max_file_size})"
                )

            # Open video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0:
                raise ValueError("Video has no frames or metadata is unreadable")

            logger.info(f"Video info: {total_frames} frames, {video_fps} FPS")

            # Calculate frame indices to extract
            if num_frames >= total_frames:
                logger.warning(
                    f"Requested {num_frames} frames, but video only has {total_frames}"
                )
                frame_indices = list(range(total_frames))
            else:
                # Extract frames evenly distributed across the video
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            # Extract frames
            extracted_frames = []
            current_frame = 0

            while cap.isOpened() and len(extracted_frames) < len(frame_indices):
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame in frame_indices:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    extracted_frames.append(frame_rgb)

                current_frame += 1

            cap.release()

            logger.info(f"Successfully extracted {len(extracted_frames)} frames")
            return extracted_frames

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            if "cap" in locals():
                cap.release()
            raise

    async def get_video_info(self, video_path: str) -> dict:
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
                raise RuntimeError(f"Could not open video file: {video_path}")

            # Extract video properties
            info = {
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": 0,
                "codec": "",
                "file_size": os.path.getsize(video_path),
            }

            # Calculate duration
            if info["fps"] > 0:
                info["duration"] = info["total_frames"] / info["fps"]

            # Get codec information (if available)
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            if fourcc:
                codec_bytes = int(fourcc).to_bytes(4, byteorder="little")
                try:
                    info["codec"] = codec_bytes.decode("ascii").strip("\x00")
                except:
                    info["codec"] = "unknown"

            cap.release()
            return info

        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
            if "cap" in locals():
                cap.release()
            raise

    async def get_video_duration(self, video_path: str) -> float:
        """
        Get video duration in seconds

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            cap.release()

            if fps > 0:
                return frame_count / fps
            return 0.0

        except Exception as e:
            logger.error(f"Error getting video duration: {e}")
            return 0.0

    def validate_video_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate video file format and properties

        Args:
            file_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file existence
            if not os.path.exists(file_path):
                return False, "File does not exist"

            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return False, f"Unsupported format: {file_ext}"

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "File is empty"

            if file_size > self.max_file_size:
                return False, f"File too large: {file_size} bytes"

            # Try to open with OpenCV
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False, "Cannot open video file"

            # Check if video has frames
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_count <= 0:
                cap.release()
                return False, "Video has no frames"

            cap.release()
            return True, "Valid video file"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def extract_frame_at_time(
        self, video_path: str, timestamp: float
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame at specific timestamp

        Args:
            video_path: Path to video file
            timestamp: Time in seconds

        Returns:
            Frame as numpy array or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            # Set position to timestamp
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            cap.release()

            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None

        except Exception as e:
            logger.error(f"Error extracting frame at {timestamp}s: {e}")
            return None

    async def create_video_thumbnail(
        self,
        video_path: str,
        output_path: str,
        timestamp: float = 1.0,
        size: Tuple[int, int] = (320, 240),
    ) -> bool:
        """
        Create a thumbnail image from video

        Args:
            video_path: Path to video file
            output_path: Path for output thumbnail
            timestamp: Time in seconds for thumbnail
            size: Thumbnail size (width, height)

        Returns:
            True if successful, False otherwise
        """
        try:
            frame = await self.extract_frame_at_time(video_path, timestamp)
            if frame is None:
                return False

            # Resize frame
            frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

            # Save thumbnail
            success = cv2.imwrite(output_path, frame_bgr)
            return success

        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return False

    def get_supported_formats(self) -> set:
        """Get set of supported video formats"""
        return self.supported_formats.copy()

    def set_max_file_size(self, size_bytes: int):
        """Set maximum allowed file size"""
        self.max_file_size = size_bytes
        logger.info(f"Max file size set to {size_bytes} bytes")

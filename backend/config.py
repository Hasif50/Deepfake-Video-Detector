"""
Configuration management for Deepfake Detector backend
Handles environment variables and application settings
From Hasif's Workspace
"""

import os
from typing import Optional
from pydantic import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application settings
    APP_NAME: str = "Deepfake Video Detector"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"

    # Model settings
    MODEL_PATH: str = os.path.join(os.getcwd(), "data", "models")
    MODEL_NAME: str = "deepfake_detector_best.pth"
    MODEL_ARCHITECTURE: str = "efficientnet_b0"

    # File handling settings
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "data", "uploads")
    OUTPUT_DIR: str = os.path.join(os.getcwd(), "data", "outputs")
    TEMP_DIR: str = os.path.join(os.getcwd(), "data", "temp")

    # Processing settings
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    MAX_FRAMES_PER_VIDEO: int = 50
    DEFAULT_FRAMES_TO_PROCESS: int = 5
    SUPPORTED_VIDEO_FORMATS: list = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

    # Model inference settings
    BATCH_SIZE: int = 8
    CONFIDENCE_THRESHOLD: float = 0.5
    ENABLE_GPU: bool = True

    # Grad-CAM settings
    GRADCAM_ENABLED: bool = True
    GRADCAM_OUTPUT_SIZE: tuple = (224, 224)
    GRADCAM_COLORMAP: str = "jet"

    # Security settings
    CORS_ORIGINS: list = ["*"]  # Configure for production
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None

    # Cache settings
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 100

    # Cleanup settings
    AUTO_CLEANUP: bool = True
    CLEANUP_INTERVAL_HOURS: int = 24
    TEMP_FILE_MAX_AGE_HOURS: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()

        # Create necessary directories
        for directory in [
            _settings.MODEL_PATH,
            _settings.UPLOAD_DIR,
            _settings.OUTPUT_DIR,
            _settings.TEMP_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

    return _settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values"""
    global _settings
    if _settings is None:
        _settings = Settings()

    for key, value in kwargs.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)

    return _settings


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""

    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8501"]


class ProductionSettings(Settings):
    """Production environment settings"""

    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    CORS_ORIGINS: list = []  # Configure specific origins
    RATE_LIMIT_REQUESTS: int = 50
    MAX_FILE_SIZE: int = 200 * 1024 * 1024  # 200MB for production


class TestingSettings(Settings):
    """Testing environment settings"""

    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    MODEL_PATH: str = os.path.join(os.getcwd(), "tests", "data", "models")
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "tests", "data", "uploads")
    OUTPUT_DIR: str = os.path.join(os.getcwd(), "tests", "data", "outputs")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB for testing


def get_settings_for_environment(env: str = None) -> Settings:
    """Get settings for specific environment"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Configuration validation
def validate_settings(settings: Settings) -> bool:
    """Validate settings configuration"""
    try:
        # Check required directories
        required_dirs = [settings.MODEL_PATH, settings.UPLOAD_DIR, settings.OUTPUT_DIR]

        for directory in required_dirs:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    print(f"Cannot create directory {directory}: {e}")
                    return False

        # Validate file size limits
        if settings.MAX_FILE_SIZE <= 0:
            print("MAX_FILE_SIZE must be positive")
            return False

        # Validate batch size
        if settings.BATCH_SIZE <= 0:
            print("BATCH_SIZE must be positive")
            return False

        # Validate confidence threshold
        if not 0 <= settings.CONFIDENCE_THRESHOLD <= 1:
            print("CONFIDENCE_THRESHOLD must be between 0 and 1")
            return False

        return True

    except Exception as e:
        print(f"Settings validation error: {e}")
        return False


# Utility functions
def get_model_path(model_name: str = None) -> str:
    """Get full path to model file"""
    settings = get_settings()
    if model_name is None:
        model_name = settings.MODEL_NAME
    return os.path.join(settings.MODEL_PATH, model_name)


def get_upload_path(filename: str) -> str:
    """Get full path for uploaded file"""
    settings = get_settings()
    return os.path.join(settings.UPLOAD_DIR, filename)


def get_output_path(filename: str) -> str:
    """Get full path for output file"""
    settings = get_settings()
    return os.path.join(settings.OUTPUT_DIR, filename)


def is_supported_video_format(filename: str) -> bool:
    """Check if video format is supported"""
    settings = get_settings()
    file_ext = Path(filename).suffix.lower()
    return file_ext in settings.SUPPORTED_VIDEO_FORMATS


# Initialize settings on import
if __name__ != "__main__":
    # Validate settings on import
    current_settings = get_settings()
    if not validate_settings(current_settings):
        print("Warning: Settings validation failed")

# Export commonly used settings
__all__ = [
    "Settings",
    "get_settings",
    "update_settings",
    "get_settings_for_environment",
    "validate_settings",
    "get_model_path",
    "get_upload_path",
    "get_output_path",
    "is_supported_video_format",
]

"""
Download Dependencies Script
Downloads required models, data, and other dependencies
From Hasif's Workspace
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import logging
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/models",
        "data/raw",
        "data/processed",
        "data/uploads",
        "data/outputs",
        "data/temp",
        "logs",
        "configs",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_file(url, filepath, description="file"):
    """Download a file with progress"""
    try:
        logger.info(f"Downloading {description}...")

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                sys.stdout.write(f"\rProgress: {percent:.1f}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, filepath, progress_hook)
        print()  # New line after progress
        logger.info(f"Downloaded {description} to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error downloading {description}: {e}")
        return False


def install_python_dependencies():
    """Install Python dependencies"""
    try:
        logger.info("Installing Python dependencies...")

        # Install main requirements
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )

        logger.info("Python dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def download_sample_data():
    """Download sample data for testing"""
    try:
        # Create sample configuration files
        sample_configs = {
            "configs/model_config.yaml": """
# Model Configuration
model:
  architecture: "efficientnet_b0"
  num_classes: 1
  pretrained: true
  dropout_rate: 0.2

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 10
  optimizer: "adam"
  
data:
  input_size: [224, 224]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
""",
            "configs/api_config.yaml": """
# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  
processing:
  max_file_size: 524288000  # 500MB
  max_frames: 50
  default_frames: 5
  
security:
  cors_origins: ["*"]
  rate_limit: 100
""",
            ".env.example": """
# Environment Variables Example
ENVIRONMENT=development
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=./data/models
UPLOAD_DIR=./data/uploads
OUTPUT_DIR=./data/outputs
""",
        }

        for filepath, content in sample_configs.items():
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Created sample config: {filepath}")

        return True

    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return False


def setup_git_hooks():
    """Setup git hooks for development"""
    try:
        if os.path.exists(".git"):
            # Create pre-commit hook
            hook_content = """#!/bin/sh
# Pre-commit hook for code quality
echo "Running pre-commit checks..."

# Check Python syntax
python -m py_compile $(find . -name "*.py" | head -10) || exit 1

echo "Pre-commit checks passed!"
"""

            hook_path = ".git/hooks/pre-commit"
            with open(hook_path, "w") as f:
                f.write(hook_content)

            # Make executable
            os.chmod(hook_path, 0o755)
            logger.info("Git pre-commit hook installed")

        return True

    except Exception as e:
        logger.warning(f"Could not setup git hooks: {e}")
        return False


def verify_installation():
    """Verify that everything is installed correctly"""
    try:
        logger.info("Verifying installation...")

        # Check Python imports
        required_modules = [
            "torch",
            "torchvision",
            "cv2",
            "numpy",
            "streamlit",
            "fastapi",
            "uvicorn",
        ]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✅ {module}")
            except ImportError:
                missing_modules.append(module)
                logger.error(f"❌ {module}")

        if missing_modules:
            logger.error(f"Missing modules: {missing_modules}")
            return False

        # Check directories
        required_dirs = ["data/models", "data/uploads", "data/outputs", "configs"]

        for directory in required_dirs:
            if os.path.exists(directory):
                logger.info(f"✅ Directory: {directory}")
            else:
                logger.error(f"❌ Directory: {directory}")
                return False

        logger.info("✅ Installation verification completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return False


def main():
    """Main function"""
    logger.info("🚀 Starting dependency download and setup...")

    success = True

    # Step 1: Create directories
    logger.info("Step 1: Creating directories...")
    create_directories()

    # Step 2: Install Python dependencies
    logger.info("Step 2: Installing Python dependencies...")
    if not install_python_dependencies():
        success = False

    # Step 3: Download sample data
    logger.info("Step 3: Setting up sample configurations...")
    if not download_sample_data():
        success = False

    # Step 4: Setup development tools
    logger.info("Step 4: Setting up development tools...")
    setup_git_hooks()

    # Step 5: Verify installation
    logger.info("Step 5: Verifying installation...")
    if not verify_installation():
        success = False

    # Final message
    if success:
        logger.info("🎉 Setup completed successfully!")
        logger.info("Next steps:")
        logger.info("  1. Run 'python run_backend.py' to start the backend")
        logger.info("  2. Run 'streamlit run frontend/app.py' to start the frontend")
        logger.info(
            "  3. Or run 'streamlit run simplified_app.py' for the simple version"
        )
    else:
        logger.error("❌ Setup completed with errors. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Backend Development Server Runner
Starts the FastAPI backend server for development
From Hasif's Workspace
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def setup_environment():
    """Setup development environment"""
    # Create necessary directories
    directories = ["data/models", "data/uploads", "data/outputs", "data/temp", "logs"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def main():
    """Main function to start the backend server"""
    logger.info("🚀 Starting Deepfake Detector Backend Server...")

    # Setup environment
    setup_environment()

    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("DEBUG", "true").lower() == "true"

    logger.info(f"Server configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Reload: {reload}")
    logger.info(f"  Environment: {'Development' if reload else 'Production'}")

    try:
        # Start the server
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

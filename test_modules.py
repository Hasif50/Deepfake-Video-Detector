"""
Module Testing Script
Tests all modules and components for functionality
From Hasif's Workspace
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all critical imports"""
    logger.info("Testing imports...")

    tests = []

    # Core dependencies
    try:
        import torch
        import torchvision

        tests.append(("PyTorch", True, torch.__version__))
    except ImportError as e:
        tests.append(("PyTorch", False, str(e)))

    try:
        import cv2

        tests.append(("OpenCV", True, cv2.__version__))
    except ImportError as e:
        tests.append(("OpenCV", False, str(e)))

    try:
        import numpy as np

        tests.append(("NumPy", True, np.__version__))
    except ImportError as e:
        tests.append(("NumPy", False, str(e)))

    try:
        import streamlit as st

        tests.append(("Streamlit", True, st.__version__))
    except ImportError as e:
        tests.append(("Streamlit", False, str(e)))

    try:
        import fastapi

        tests.append(("FastAPI", True, fastapi.__version__))
    except ImportError as e:
        tests.append(("FastAPI", False, str(e)))

    try:
        import uvicorn

        tests.append(("Uvicorn", True, uvicorn.__version__))
    except ImportError as e:
        tests.append(("Uvicorn", False, str(e)))

    # Optional dependencies
    try:
        from pytorch_grad_cam import GradCAM

        tests.append(("Grad-CAM", True, "Available"))
    except ImportError as e:
        tests.append(("Grad-CAM", False, "Not available"))

    # Print results
    logger.info("Import test results:")
    all_passed = True
    for name, passed, version in tests:
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {name}: {version}")
        if not passed and name in [
            "PyTorch",
            "OpenCV",
            "NumPy",
            "Streamlit",
            "FastAPI",
        ]:
            all_passed = False

    return all_passed


def test_model_creation():
    """Test model creation and basic operations"""
    logger.info("Testing model creation...")

    try:
        # Test simple model creation
        from models.deepfake_detector import DeepfakeDetector

        model = DeepfakeDetector(num_classes=1, pretrained=False)
        logger.info("✅ Model creation successful")

        # Test forward pass
        import torch

        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)

        if output.shape == (1, 1):
            logger.info("✅ Model forward pass successful")
            return True
        else:
            logger.error(f"❌ Unexpected output shape: {output.shape}")
            return False

    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False


def test_video_processing():
    """Test video processing capabilities"""
    logger.info("Testing video processing...")

    try:
        import cv2
        import numpy as np

        # Create a dummy video frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test basic OpenCV operations
        gray = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(dummy_frame, (224, 224))

        logger.info("✅ Video processing operations successful")
        return True

    except Exception as e:
        logger.error(f"❌ Video processing test failed: {e}")
        return False


def test_backend_components():
    """Test backend components"""
    logger.info("Testing backend components...")

    try:
        # Test configuration
        sys.path.insert(0, "backend")
        from config import get_settings

        settings = get_settings()
        logger.info("✅ Configuration loading successful")

        # Test model handler (without actual model loading)
        from model_handler import ModelHandler

        handler = ModelHandler()
        logger.info("✅ Model handler initialization successful")

        return True

    except Exception as e:
        logger.error(f"❌ Backend component test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_directory_structure():
    """Test directory structure"""
    logger.info("Testing directory structure...")

    required_dirs = ["src", "backend", "frontend", "data", "docs"]

    optional_dirs = ["data/models", "data/uploads", "data/outputs", "configs", "tests"]

    all_good = True

    for directory in required_dirs:
        if os.path.exists(directory):
            logger.info(f"✅ Required directory: {directory}")
        else:
            logger.error(f"❌ Missing required directory: {directory}")
            all_good = False

    for directory in optional_dirs:
        if os.path.exists(directory):
            logger.info(f"✅ Optional directory: {directory}")
        else:
            logger.warning(f"⚠️ Optional directory missing: {directory}")

    return all_good


def test_file_structure():
    """Test critical files"""
    logger.info("Testing file structure...")

    critical_files = [
        "README.md",
        "requirements.txt",
        "docker-compose.yml",
        "backend/main.py",
        "frontend/app.py",
        "simplified_app.py",
    ]

    all_good = True

    for filepath in critical_files:
        if os.path.exists(filepath):
            logger.info(f"✅ Critical file: {filepath}")
        else:
            logger.error(f"❌ Missing critical file: {filepath}")
            all_good = False

    return all_good


def test_gpu_availability():
    """Test GPU availability"""
    logger.info("Testing GPU availability...")

    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU available: {device_name} (Count: {device_count})")
        else:
            logger.info("ℹ️ GPU not available, will use CPU")

        return True

    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("🧪 Starting comprehensive module testing...")

    tests = [
        ("Import Tests", test_imports),
        ("Directory Structure", test_directory_structure),
        ("File Structure", test_file_structure),
        ("Model Creation", test_model_creation),
        ("Video Processing", test_video_processing),
        ("Backend Components", test_backend_components),
        ("GPU Availability", test_gpu_availability),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ ERROR - {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Please check the issues above.")
        return False


def main():
    """Main function"""
    try:
        success = run_all_tests()

        if success:
            logger.info("\n🚀 System is ready for use!")
            logger.info("Next steps:")
            logger.info("  1. Run 'python run_backend.py' to start backend")
            logger.info("  2. Run 'streamlit run frontend/app.py' for full app")
            logger.info("  3. Run 'streamlit run simplified_app.py' for simple app")
        else:
            logger.error(
                "\n❌ System has issues. Please resolve them before proceeding."
            )
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nTesting interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

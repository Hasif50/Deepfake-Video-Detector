#!/usr/bin/env python3
"""
Project Verification Script
Verifies that the project is ready for main branch deployment
From Hasif's Workspace
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath, description=""):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description or filepath}")
        return True
    else:
        print(f"❌ Missing: {description or filepath}")
        return False


def check_directory_exists(dirpath, description=""):
    """Check if a directory exists"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description or dirpath}")
        return True
    else:
        print(f"❌ Missing directory: {description or dirpath}")
        return False


def verify_project_structure():
    """Verify the complete project structure"""
    print("🔍 Verifying Project Structure...")
    print("=" * 50)

    all_good = True

    # Core files
    core_files = [
        ("README.md", "Main documentation"),
        ("SIMPLIFIED_README.md", "Quick start guide"),
        ("requirements.txt", "Python dependencies"),
        ("docker-compose.yml", "Docker orchestration"),
        ("Dockerfile", "Main container"),
        (".gitignore", "Git ignore rules"),
        (".gitattributes", "Git attributes"),
        ("LICENSE", "MIT License"),
        ("PROJECT_INFO.md", "Project information"),
        (".env.example", "Environment variables example"),
    ]

    for filepath, description in core_files:
        if not check_file_exists(filepath, description):
            all_good = False

    # Executable scripts
    scripts = [
        ("run_backend.py", "Backend runner"),
        ("simplified_app.py", "Simplified app"),
        ("download_dependencies.py", "Dependency downloader"),
        ("test_modules.py", "Module tester"),
        ("verify_project.py", "Project verifier"),
    ]

    for filepath, description in scripts:
        if not check_file_exists(filepath, description):
            all_good = False

    # Directories
    directories = [
        ("backend", "FastAPI backend"),
        ("frontend", "Streamlit frontend"),
        ("src", "Source code"),
        ("src/models", "Model implementations"),
        ("src/data", "Data processing"),
        ("src/training", "Training pipeline"),
        ("tests", "Test suite"),
        ("docs", "Documentation"),
        ("configs", "Configuration files"),
    ]

    for dirpath, description in directories:
        if not check_directory_exists(dirpath, description):
            all_good = False

    # Backend files
    backend_files = [
        ("backend/main.py", "FastAPI main"),
        ("backend/model_handler.py", "Model handler"),
        ("backend/video_processor.py", "Video processor"),
        ("backend/explainability_engine.py", "XAI engine"),
        ("backend/config.py", "Backend config"),
        ("backend/requirements.txt", "Backend dependencies"),
        ("backend/Dockerfile", "Backend container"),
    ]

    for filepath, description in backend_files:
        if not check_file_exists(filepath, description):
            all_good = False

    # Frontend files
    frontend_files = [
        ("frontend/app.py", "Streamlit app"),
        ("frontend/requirements.txt", "Frontend dependencies"),
        ("frontend/Dockerfile", "Frontend container"),
    ]

    for filepath, description in frontend_files:
        if not check_file_exists(filepath, description):
            all_good = False

    # Source files
    src_files = [
        ("src/__init__.py", "Source package init"),
        ("src/models/__init__.py", "Models package"),
        ("src/models/deepfake_detector.py", "Main model"),
        ("src/models/model_utils.py", "Model utilities"),
        ("src/data/__init__.py", "Data package"),
        ("src/data/preprocessor.py", "Video preprocessor"),
        ("src/data/dataset.py", "PyTorch datasets"),
        ("src/data/augmentation.py", "Data augmentation"),
        ("src/training/__init__.py", "Training package"),
        ("src/training/trainer.py", "Model trainer"),
        ("src/training/losses.py", "Loss functions"),
        ("src/training/metrics.py", "Evaluation metrics"),
    ]

    for filepath, description in src_files:
        if not check_file_exists(filepath, description):
            all_good = False

    # Documentation files
    doc_files = [
        ("docs/api_documentation.md", "API documentation"),
        ("docs/evaluation_strategy.md", "Evaluation strategy"),
        ("docs/optimization_strategies.md", "Optimization strategies"),
    ]

    for filepath, description in doc_files:
        if not check_file_exists(filepath, description):
            all_good = False

    # Configuration files
    config_files = [
        ("configs/model_config.yaml", "Model configuration"),
        ("configs/api_config.yaml", "API configuration"),
    ]

    for filepath, description in config_files:
        if not check_file_exists(filepath, description):
            all_good = False

    # Test files
    test_files = [
        ("tests/__init__.py", "Test package"),
        ("tests/test_backend.py", "Backend tests"),
    ]

    for filepath, description in test_files:
        if not check_file_exists(filepath, description):
            all_good = False

    return all_good


def verify_content_attribution():
    """Verify that files contain proper attribution"""
    print("\n🔍 Verifying Content Attribution...")
    print("=" * 50)

    # Files that should contain "Hasif's Workspace"
    files_to_check = [
        "README.md",
        "SIMPLIFIED_README.md",
        "src/__init__.py",
        "backend/main.py",
        "frontend/app.py",
        "simplified_app.py",
        "PROJECT_INFO.md",
    ]

    all_good = True

    for filepath in files_to_check:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "Hasif's Workspace" in content or "Hasif" in content:
                        print(f"✅ {filepath} - Contains proper attribution")
                    else:
                        print(f"⚠️  {filepath} - Missing attribution")
                        all_good = False
            except Exception as e:
                print(f"❌ Error reading {filepath}: {e}")
                all_good = False
        else:
            print(f"❌ File not found: {filepath}")
            all_good = False

    return all_good


def verify_no_unwanted_references():
    """Verify no unwanted references exist"""
    print("\n🔍 Checking for Unwanted References...")
    print("=" * 50)

    # Check that sample files were removed
    unwanted_files = ["sample_resume.txt", "sample_job_description.txt"]

    all_good = True

    for filepath in unwanted_files:
        if os.path.exists(filepath):
            print(f"❌ Unwanted file still exists: {filepath}")
            all_good = False
        else:
            print(f"✅ Unwanted file removed: {filepath}")

    return all_good


def verify_repository_readiness():
    """Verify repository is ready for main branch"""
    print("\n🔍 Verifying Repository Readiness...")
    print("=" * 50)

    # Check for correct repository references
    files_with_repo_refs = ["README.md", "SIMPLIFIED_README.md"]

    all_good = True

    for filepath in files_with_repo_refs:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "Deepfake-Video-Detector" in content:
                        print(f"✅ {filepath} - Contains correct repository name")
                    else:
                        print(f"⚠️  {filepath} - Check repository references")
                        all_good = False
            except Exception as e:
                print(f"❌ Error reading {filepath}: {e}")
                all_good = False

    return all_good


def main():
    """Main verification function"""
    print("🚀 Deepfake Video Detector - Project Verification")
    print("From Hasif's Workspace")
    print("=" * 60)

    # Run all verifications
    structure_ok = verify_project_structure()
    attribution_ok = verify_content_attribution()
    no_unwanted_ok = verify_no_unwanted_references()
    repo_ready_ok = verify_repository_readiness()

    # Final summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)

    if structure_ok:
        print("✅ Project Structure: COMPLETE")
    else:
        print("❌ Project Structure: INCOMPLETE")

    if attribution_ok:
        print("✅ Content Attribution: CORRECT")
    else:
        print("❌ Content Attribution: MISSING")

    if no_unwanted_ok:
        print("✅ Unwanted Files: REMOVED")
    else:
        print("❌ Unwanted Files: STILL PRESENT")

    if repo_ready_ok:
        print("✅ Repository References: CORRECT")
    else:
        print("❌ Repository References: NEED UPDATE")

    overall_status = all([structure_ok, attribution_ok, no_unwanted_ok, repo_ready_ok])

    print("\n" + "=" * 60)
    if overall_status:
        print("🎉 PROJECT READY FOR MAIN BRANCH DEPLOYMENT!")
        print("✅ All verifications passed")
        print("🚀 Ready to push to: https://github.com/Hasif50/Deepfake-Video-Detector")
    else:
        print("⚠️  PROJECT NEEDS ATTENTION")
        print("❌ Some verifications failed")
        print("🔧 Please fix the issues above before deployment")

    print("=" * 60)

    return overall_status


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

@echo off
REM Git Setup Script for Deepfake Video Detector (Windows)
REM From Hasif's Workspace

echo.
echo 🚀 Deepfake Video Detector - Git Setup (Windows)
echo From Hasif's Workspace
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed. Please install Git first.
    pause
    exit /b 1
)
echo ✅ Git is installed

REM Check if we're in the right directory
if not exist "README.md" (
    echo ❌ README.md not found. Please run this script from the project root directory
    pause
    exit /b 1
)
if not exist "PROJECT_INFO.md" (
    echo ❌ PROJECT_INFO.md not found. Please run this script from the project root directory
    pause
    exit /b 1
)
echo ✅ In correct project directory

REM Run project verification
echo.
echo 🔍 Running project verification...
python verify_project.py
if errorlevel 1 (
    echo ❌ Project verification failed. Please fix issues before proceeding.
    pause
    exit /b 1
)
echo ✅ Project verification passed

REM Check if git repository already exists
if exist ".git" (
    echo ⚠️  Git repository already exists
    set /p "reinit=Do you want to reinitialize? (y/N): "
    if /i "%reinit%"=="y" (
        rmdir /s /q .git
        echo ✅ Removed existing git repository
    ) else (
        echo ℹ️  Keeping existing git repository
    )
)

REM Initialize git repository
if not exist ".git" (
    echo.
    echo 📁 Initializing Git repository...
    git init
    echo ✅ Git repository initialized
)

REM Configure git user (optional)
echo.
echo 👤 Git Configuration
echo Current git user configuration:
git config user.name 2>nul || echo   Name: Not set
git config user.email 2>nul || echo   Email: Not set
echo.
set /p "config_user=Do you want to set/update git user info for this repository? (y/N): "
if /i "%config_user%"=="y" (
    set /p "git_name=Enter your name: "
    set /p "git_email=Enter your email: "
    git config user.name "%git_name%"
    git config user.email "%git_email%"
    echo ✅ Git user configuration updated
)

REM Add all files
echo.
echo 📦 Adding files to git...
git add .
echo ✅ All files added to staging area

REM Create initial commit
echo.
echo 💾 Creating initial commit...
git commit -m "Initial commit: Deepfake Video Detector

- Complete AI-powered deepfake detection system
- FastAPI backend with REST API endpoints
- Streamlit frontend with interactive UI
- Advanced model architecture with EfficientNet-B0
- Grad-CAM explainable AI visualizations
- Comprehensive testing suite
- Docker containerization support
- Production-ready deployment configuration

From Hasif's Workspace

Features:
- Video upload and processing
- Frame-by-frame analysis
- Real-time deepfake detection
- Visual explanations with Grad-CAM
- RESTful API with OpenAPI documentation
- Modern web interface
- Containerized deployment
- Comprehensive test coverage

Technology Stack:
- PyTorch for deep learning
- FastAPI for backend API
- Streamlit for frontend
- OpenCV for video processing
- Docker for containerization
- pytest for testing"

if errorlevel 1 (
    echo ❌ Failed to create initial commit
    pause
    exit /b 1
)
echo ✅ Initial commit created successfully

REM Add remote repository
echo.
echo 🌐 Setting up remote repository...
set REPO_URL=https://github.com/Hasif50/Deepfake-Video-Detector.git

REM Check if remote already exists
git remote get-url origin >nul 2>&1
if not errorlevel 1 (
    echo ⚠️  Remote 'origin' already exists
    for /f "tokens=*" %%i in ('git remote get-url origin') do set current_url=%%i
    echo Current URL: %current_url%
    
    if not "%current_url%"=="%REPO_URL%" (
        set /p "update_remote=Do you want to update the remote URL? (y/N): "
        if /i "!update_remote!"=="y" (
            git remote set-url origin "%REPO_URL%"
            echo ✅ Remote URL updated
        )
    ) else (
        echo ✅ Remote URL is already correct
    )
) else (
    git remote add origin "%REPO_URL%"
    echo ✅ Remote repository added
)

REM Set main branch
echo.
echo 🌿 Setting up main branch...
git branch -M main
echo ✅ Main branch configured

REM Show status
echo.
echo 📊 Repository Status:
echo ====================
git status --short
echo.
git log --oneline -1

REM Final instructions
echo.
echo 🎯 Setup Complete!
echo ==================
echo ✅ Git repository is ready for deployment
echo ℹ️  Repository URL: %REPO_URL%
echo ℹ️  Branch: main
echo.
echo 🚀 Next Steps:
echo ==============
echo 1. Push to GitHub:
echo    git push -u origin main
echo.
echo 2. Verify deployment:
echo    Visit: https://github.com/Hasif50/Deepfake-Video-Detector
echo.
echo 3. Set repository description:
echo    'AI-powered deepfake video detection system with explainable AI'
echo.
echo 4. Add topics:
echo    deepfake-detection, computer-vision, pytorch, fastapi, streamlit, explainable-ai

REM Ask if user wants to push now
echo.
set /p "push_now=Do you want to push to GitHub now? (y/N): "
if /i "%push_now%"=="y" (
    echo.
    echo 🚀 Pushing to GitHub...
    git push -u origin main
    if not errorlevel 1 (
        echo ✅ Successfully pushed to GitHub!
        echo ℹ️  Repository is now live at: https://github.com/Hasif50/Deepfake-Video-Detector
    ) else (
        echo ❌ Failed to push to GitHub
        echo ℹ️  You may need to authenticate or check your internet connection
        echo ℹ️  Try running: git push -u origin main
    )
) else (
    echo ℹ️  You can push later using: git push -u origin main
)

echo.
echo ✨ Deployment setup complete!
echo From Hasif's Workspace with ❤️
echo.
pause

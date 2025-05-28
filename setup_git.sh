#!/bin/bash

# Git Setup Script for Deepfake Video Detector
# From Hasif's Workspace

echo "🚀 Deepfake Video Detector - Git Setup"
echo "From Hasif's Workspace"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

print_status "Git is installed"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "PROJECT_INFO.md" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_status "In correct project directory"

# Run project verification
echo ""
echo "🔍 Running project verification..."
if python verify_project.py; then
    print_status "Project verification passed"
else
    print_error "Project verification failed. Please fix issues before proceeding."
    exit 1
fi

# Check if git repository already exists
if [ -d ".git" ]; then
    print_warning "Git repository already exists"
    read -p "Do you want to reinitialize? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .git
        print_status "Removed existing git repository"
    else
        print_info "Keeping existing git repository"
    fi
fi

# Initialize git repository
if [ ! -d ".git" ]; then
    echo ""
    echo "📁 Initializing Git repository..."
    git init
    print_status "Git repository initialized"
fi

# Configure git user (optional)
echo ""
echo "👤 Git Configuration"
echo "Current git user configuration:"
git config user.name 2>/dev/null || echo "  Name: Not set"
git config user.email 2>/dev/null || echo "  Email: Not set"

read -p "Do you want to set/update git user info for this repository? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your name: " git_name
    read -p "Enter your email: " git_email
    
    git config user.name "$git_name"
    git config user.email "$git_email"
    print_status "Git user configuration updated"
fi

# Add all files
echo ""
echo "📦 Adding files to git..."
git add .
print_status "All files added to staging area"

# Create initial commit
echo ""
echo "💾 Creating initial commit..."
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

if [ $? -eq 0 ]; then
    print_status "Initial commit created successfully"
else
    print_error "Failed to create initial commit"
    exit 1
fi

# Add remote repository
echo ""
echo "🌐 Setting up remote repository..."
REPO_URL="https://github.com/Hasif50/Deepfake-Video-Detector.git"

# Check if remote already exists
if git remote get-url origin &> /dev/null; then
    print_warning "Remote 'origin' already exists"
    current_url=$(git remote get-url origin)
    echo "Current URL: $current_url"
    
    if [ "$current_url" != "$REPO_URL" ]; then
        read -p "Do you want to update the remote URL? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git remote set-url origin "$REPO_URL"
            print_status "Remote URL updated"
        fi
    else
        print_status "Remote URL is already correct"
    fi
else
    git remote add origin "$REPO_URL"
    print_status "Remote repository added"
fi

# Set main branch
echo ""
echo "🌿 Setting up main branch..."
git branch -M main
print_status "Main branch configured"

# Show status
echo ""
echo "📊 Repository Status:"
echo "===================="
git status --short
echo ""
git log --oneline -1

# Final instructions
echo ""
echo "🎯 Setup Complete!"
echo "=================="
print_status "Git repository is ready for deployment"
print_info "Repository URL: $REPO_URL"
print_info "Branch: main"

echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Push to GitHub:"
echo "   git push -u origin main"
echo ""
echo "2. Verify deployment:"
echo "   Visit: https://github.com/Hasif50/Deepfake-Video-Detector"
echo ""
echo "3. Set repository description:"
echo "   'AI-powered deepfake video detection system with explainable AI'"
echo ""
echo "4. Add topics:"
echo "   deepfake-detection, computer-vision, pytorch, fastapi, streamlit, explainable-ai"

# Ask if user wants to push now
echo ""
read -p "Do you want to push to GitHub now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Pushing to GitHub..."
    if git push -u origin main; then
        print_status "Successfully pushed to GitHub!"
        print_info "Repository is now live at: https://github.com/Hasif50/Deepfake-Video-Detector"
    else
        print_error "Failed to push to GitHub"
        print_info "You may need to authenticate or check your internet connection"
        print_info "Try running: git push -u origin main"
    fi
else
    print_info "You can push later using: git push -u origin main"
fi

echo ""
echo "✨ Deployment setup complete!"
echo "From Hasif's Workspace with ❤️"

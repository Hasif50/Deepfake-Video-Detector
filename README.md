# Deepfake Video Detector

A comprehensive AI-powered deepfake video detection system using PyTorch and EfficientNet-B0. Features FastAPI backend, Streamlit frontend, Grad-CAM visualizations for model transparency, and containerized deployment. Created from Hasif's Workspace to demonstrate expertise in computer vision, explainable AI, and solving media authenticity challenges.

## ✨ Features

- **Advanced Deepfake Detection:** Classifies videos as "Real" or "Deepfake" using state-of-the-art EfficientNet-B0 architecture
- **Frame-Level Analysis:** Processes individual video frames with sophisticated preprocessing
- **Video-Level Aggregation:** Intelligent aggregation of frame predictions for robust video classification
- **Explainable AI (XAI):** Grad-CAM visualizations highlighting decision-making regions
- **FastAPI Backend:** High-performance REST API for scalable inference
- **Streamlit Frontend:** User-friendly web interface for video upload and analysis
- **Real-time Processing:** Efficient video processing pipeline with optimized frame extraction
- **Containerized Deployment:** Docker support for easy deployment and scaling
- **Comprehensive Testing:** Full test suite ensuring reliability
- **Modular Architecture:** Clean, maintainable codebase with separation of concerns

## 🛠️ Tech Stack

### **Backend:**
- **Python 3.10+**
- **FastAPI:** High-performance web framework for building APIs
- **PyTorch:** Deep learning framework for model inference
- **TorchVision:** Computer vision utilities and pre-trained models
- **OpenCV:** Video processing and computer vision operations
- **Grad-CAM:** Explainable AI visualizations
- **Uvicorn:** ASGI server for FastAPI
- **NumPy & Scikit-learn:** Scientific computing and machine learning utilities

### **Frontend:**
- **Streamlit:** Interactive web application framework
- **Requests:** HTTP client for backend communication
- **Pillow:** Image processing and manipulation

### **Model & AI:**
- **EfficientNet-B0:** Pre-trained CNN backbone for feature extraction
- **Transfer Learning:** Fine-tuned on deepfake detection datasets
- **Grad-CAM:** Gradient-weighted Class Activation Mapping for explainability

### **Infrastructure:**
- **Docker & Docker Compose:** Containerization and orchestration
- **Git:** Version control with comprehensive .gitignore

## 📂 Project Structure

```
deepfake-detector/
├── backend/                    # FastAPI backend application
│   ├── main.py                # FastAPI app definition and endpoints
│   ├── model_handler.py       # Model loading and inference logic
│   ├── video_processor.py     # Video processing utilities
│   ├── frame_extractor.py     # Frame extraction from videos
│   ├── explainability_engine.py # Grad-CAM and XAI functionality
│   ├── config.py              # Configuration management
│   ├── requirements.txt       # Backend dependencies
│   └── Dockerfile             # Backend container configuration
├── frontend/                   # Streamlit frontend application
│   ├── app.py                 # Main Streamlit application
│   ├── components/            # Reusable UI components
│   ├── utils/                 # Frontend utilities
│   ├── requirements.txt       # Frontend dependencies
│   └── Dockerfile             # Frontend container configuration
├── src/                       # Core source code modules
│   ├── models/                # Model definitions and utilities
│   │   ├── __init__.py
│   │   ├── deepfake_detector.py # Main model architecture
│   │   └── model_utils.py     # Model utility functions
│   ├── data/                  # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessor.py    # Data preprocessing pipeline
│   │   ├── dataset.py         # PyTorch dataset implementations
│   │   └── augmentation.py    # Data augmentation strategies
│   ├── training/              # Training scripts and utilities
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training orchestration
│   │   ├── losses.py          # Custom loss functions
│   │   └── metrics.py         # Evaluation metrics
│   ├── evaluation/            # Model evaluation and testing
│   │   ├── __init__.py
│   │   ├── evaluator.py       # Model evaluation pipeline
│   │   └── robustness.py      # Robustness testing
│   └── utils/                 # Common utilities
│       ├── __init__.py
│       ├── logger.py          # Logging configuration
│       ├── config_loader.py   # Configuration loading
│       └── helpers.py         # Helper functions
├── tests/                     # Comprehensive test suite
│   ├── __init__.py
│   ├── test_backend.py        # Backend API tests
│   ├── test_models.py         # Model functionality tests
│   ├── test_preprocessing.py  # Data processing tests
│   └── test_integration.py    # Integration tests
├── data/                      # Data storage directories
│   ├── raw/                   # Raw video datasets
│   ├── processed/             # Processed frames and features
│   ├── models/                # Trained model checkpoints
│   └── outputs/               # Analysis outputs and visualizations
├── docs/                      # Enhanced documentation
│   ├── api_documentation.md   # API endpoint documentation
│   ├── model_architecture.md  # Detailed model documentation
│   ├── evaluation_strategy.md # Evaluation methodology
│   ├── optimization_strategies.md # Performance optimization
│   └── deployment_guide.md    # Deployment instructions
├── configs/                   # Configuration files
│   ├── model_config.yaml      # Model configuration
│   ├── training_config.yaml   # Training parameters
│   └── api_config.yaml        # API configuration
├── scripts/                   # Utility scripts
│   ├── download_models.py     # Model download script
│   ├── setup_environment.py   # Environment setup
│   └── run_evaluation.py      # Evaluation runner
├── docker-compose.yml         # Multi-service Docker orchestration
├── simple-docker-compose.yml  # Simplified Docker setup
├── Dockerfile                 # Main application container
├── requirements.txt           # Main project dependencies
├── run_backend.py             # Backend development server
├── simplified_app.py          # Single-file application version
├── test_modules.py            # Module testing script
├── download_dependencies.py   # Dependency management
├── .gitignore                 # Git ignore patterns
├── LICENSE                    # MIT License
└── README.md                  # This comprehensive documentation
```

## ⚙️ Setup and Installation

### Prerequisites

- **Python 3.10+**
- **pip** (Python package installer)
- **Docker and Docker Compose** (recommended for easiest setup)
- **Git** for version control

### Option 1: Using Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hasif50/Deepfake-Video-Detector.git
   cd Deepfake-Video-Detector
   ```

2. **Build and start services:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - **Frontend:** http://localhost:8501
   - **Backend API:** http://localhost:8000
   - **API Documentation:** http://localhost:8000/docs

### Option 2: Local Development

1. **Clone and setup:**
   ```bash
   git clone https://github.com/Hasif50/Deepfake-Video-Detector.git
   cd Deepfake-Video-Detector
   ```

2. **Setup backend:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cd ..
   ```

3. **Setup frontend:**
   ```bash
   cd frontend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cd ..
   ```

4. **Download dependencies:**
   ```bash
   python download_dependencies.py
   ```

5. **Start backend server:**
   ```bash
   python run_backend.py
   ```

6. **Start frontend (new terminal):**
   ```bash
   streamlit run frontend/app.py
   ```

## 📖 User Guide

### Video Upload and Analysis

1. **Access the application** at http://localhost:8501
2. **Upload your video file** (supported formats: MP4, AVI, MOV)
3. **Configure analysis parameters:**
   - Number of frames to process
   - Grad-CAM visualization options
   - Confidence thresholds
4. **Click "Analyze Video"** to start processing
5. **Review results:**
   - Overall authenticity score
   - Frame-by-frame analysis
   - Grad-CAM heatmaps showing decision regions
   - Detailed confidence metrics

### API Usage

The backend provides comprehensive REST API endpoints:

#### Analyze Video Endpoint
```bash
POST /api/v1/analyze-video
Content-Type: multipart/form-data

# Upload video file and get analysis results
curl -X POST "http://localhost:8000/api/v1/analyze-video" \
     -F "video_file=@your_video.mp4" \
     -F "num_frames=10" \
     -F "enable_gradcam=true"
```

#### Health Check
```bash
GET /api/v1/health
# Returns system status and model information
```

### Response Format
```json
{
  "video_id": "unique_identifier",
  "overall_prediction": "Real",
  "confidence_score": 0.87,
  "frame_predictions": [
    {
      "frame_number": 1,
      "prediction": "Real",
      "confidence": 0.89,
      "gradcam_available": true
    }
  ],
  "processing_time": 2.34,
  "model_version": "efficientnet_b0_v1.0"
}
```

## 🔬 Model Architecture

The deepfake detection system uses a sophisticated architecture:

- **Backbone:** EfficientNet-B0 pre-trained on ImageNet
- **Transfer Learning:** Fine-tuned classifier for binary classification
- **Input Processing:** 224x224 RGB frames with ImageNet normalization
- **Output:** Single logit with sigmoid activation for probability
- **Aggregation:** Video-level prediction through frame averaging

## 🧠 Explainable AI

Grad-CAM (Gradient-weighted Class Activation Mapping) provides insights into model decisions:

- **Heatmap Generation:** Highlights important regions for classification
- **Layer Targeting:** Focuses on final convolutional layers
- **Visualization:** Overlays heatmaps on original frames
- **Interpretation:** Red regions indicate higher importance for decision

## 📊 Evaluation Metrics

The model is evaluated using comprehensive metrics:

- **Accuracy:** Overall classification correctness
- **Precision:** True positive rate for deepfake detection
- **Recall:** Sensitivity to actual deepfakes
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under the receiver operating characteristic curve
- **Confusion Matrix:** Detailed breakdown of predictions

## 🚀 Performance Optimization

- **Efficient Frame Extraction:** Optimized OpenCV operations
- **Batch Processing:** Vectorized operations for multiple frames
- **GPU Acceleration:** CUDA support for faster inference
- **Model Quantization:** Reduced precision for deployment
- **Caching:** Intelligent caching of processed frames

## 🔒 Security Considerations

- **Input Validation:** Comprehensive file type and size validation
- **Rate Limiting:** API rate limiting to prevent abuse
- **Sanitization:** Secure file handling and processing
- **Error Handling:** Graceful error handling without information leakage

## 📈 Future Enhancements

- **Temporal Modeling:** LSTM/Transformer integration for sequence analysis
- **Multi-Modal Detection:** Audio analysis for comprehensive detection
- **Real-time Processing:** Live video stream analysis
- **Advanced Architectures:** Vision Transformers and newer models
- **Federated Learning:** Privacy-preserving distributed training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project was developed entirely from Hasif's Workspace, utilizing:
- **PyTorch** deep learning framework
- **EfficientNet** architecture concepts
- **Grad-CAM** explainability techniques
- **Open Source Libraries** for supporting functionality

## 👨‍💻 Author

**Mohd Hasif**
- GitHub: [@Hasif50](https://github.com/Hasif50)
- Email: hashifu50@gmail.com
- Workspace: Hasif's Workspace

---

**Note:** This project is for educational and research purposes. The effectiveness of deepfake detection depends on training data quality and model tuning. Always verify results with multiple detection methods for critical applications.

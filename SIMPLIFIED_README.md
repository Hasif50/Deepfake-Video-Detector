# Deepfake Video Detector - Quick Start Guide

A simple AI-powered tool to detect deepfake videos using deep learning. From Hasif's Workspace.

## 🚀 Quick Start (Docker - Recommended)

1. **Clone and run:**
   ```bash
   git clone https://github.com/Hasif50/Deepfake-Video-Detector.git
   cd Deepfake-Video-Detector
   docker-compose up --build
   ```

2. **Access the app:**
   - Open http://localhost:8501 in your browser
   - Upload a video file
   - Click "Analyze Video"
   - View results and explanations

## 🛠️ Manual Setup

1. **Install Python 3.10+**

2. **Quick install:**
   ```bash
   git clone https://github.com/Hasif50/Deepfake-Video-Detector.git
   cd Deepfake-Video-Detector
   pip install -r requirements.txt
   python download_dependencies.py
   ```

3. **Run the app:**
   ```bash
   streamlit run simplified_app.py
   ```

## 📱 How to Use

1. **Upload Video:** Drag and drop or browse for video file (MP4, AVI, MOV)
2. **Configure:** Set number of frames to analyze (default: 5)
3. **Analyze:** Click the analyze button
4. **Results:** View prediction, confidence score, and visual explanations

## 🎯 What You Get

- **Prediction:** Real or Deepfake classification
- **Confidence:** How certain the model is (0-100%)
- **Heatmaps:** Visual explanation of what the AI focused on
- **Frame Analysis:** Individual frame predictions

## 🔧 Troubleshooting

**Common Issues:**

1. **"Module not found" errors:**
   ```bash
   pip install -r requirements.txt
   python download_dependencies.py
   ```

2. **Video won't upload:**
   - Check file format (MP4, AVI, MOV supported)
   - Ensure file size < 200MB

3. **Slow processing:**
   - Reduce number of frames to analyze
   - Use smaller video files for testing

4. **Docker issues:**
   ```bash
   docker-compose down
   docker-compose up --build
   ```

## 📊 Understanding Results

- **Real (Green):** Video appears authentic
- **Deepfake (Red):** Video appears manipulated
- **Confidence:** Higher percentages mean more certain predictions
- **Heatmaps:** Red areas show where the AI detected suspicious patterns

## 🎓 Educational Use

This tool is designed for:
- Learning about AI and deepfake detection
- Understanding explainable AI concepts
- Demonstrating computer vision applications
- Research and educational purposes

**Important:** This is a demonstration tool. For critical applications, use multiple detection methods and expert verification.

## 📞 Support

- Check the full [README.md](README.md) for detailed documentation
- Review [docs/](docs/) folder for technical details
- Open an issue on GitHub for bugs or questions

## 🏃‍♂️ Next Steps

After trying the basic version:
1. Explore the full API at http://localhost:8000/docs
2. Try different video types and qualities
3. Experiment with analysis parameters
4. Review the technical documentation for deeper understanding

---

**Quick tip:** Start with short videos (< 30 seconds) for faster processing and better learning experience.

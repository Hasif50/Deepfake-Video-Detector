"""
Streamlit Frontend for Deepfake Video Detector
Enhanced user interface with comprehensive features
From Hasif's Workspace
"""

import streamlit as st
import requests
import os
import time
import json
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile

# Page configuration
st.set_page_config(
    page_title="Deepfake Video Detector",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
SUPPORTED_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .result-real {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .result-fake {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_backend_health() -> bool:
    """Check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_and_analyze_video(
    video_file, num_frames: int, enable_gradcam: bool, confidence_threshold: float
) -> Optional[Dict[Any, Any]]:
    """Upload video and get analysis results"""
    try:
        # Prepare files and data
        files = {"video_file": video_file}
        data = {
            "num_frames": num_frames,
            "enable_gradcam": enable_gradcam,
            "confidence_threshold": confidence_threshold,
        }

        # Make request to backend
        with st.spinner("Analyzing video... This may take a few moments."):
            response = requests.post(
                f"{BACKEND_URL}/api/v1/analyze-video",
                files=files,
                data=data,
                timeout=300,  # 5 minutes timeout
            )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.text}")
            return None

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try with a shorter video or fewer frames.")
        return None
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None


def display_results(results: Dict[Any, Any]):
    """Display analysis results in an organized manner"""

    # Main result
    overall_prediction = results["overall_prediction"]
    confidence_score = results["confidence_score"]

    if overall_prediction == "Real":
        st.markdown(
            f"""
        <div class="result-real">
            <h2>✅ Video appears to be REAL</h2>
            <h3>Confidence: {confidence_score:.1%}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="result-fake">
            <h2>⚠️ Video appears to be DEEPFAKE</h2>
            <h3>Confidence: {confidence_score:.1%}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Prediction", overall_prediction)

    with col2:
        st.metric("Confidence Score", f"{confidence_score:.1%}")

    with col3:
        st.metric("Frames Analyzed", len(results["frame_predictions"]))

    with col4:
        st.metric("Processing Time", f"{results['processing_time']:.1f}s")

    # Detailed frame analysis
    st.subheader("📊 Frame-by-Frame Analysis")

    frame_data = []
    for frame_pred in results["frame_predictions"]:
        frame_data.append(
            {
                "Frame": frame_pred["frame_number"],
                "Prediction": frame_pred["prediction"],
                "Confidence": f"{frame_pred['confidence']:.1%}",
                "Grad-CAM": "✅" if frame_pred["gradcam_available"] else "❌",
            }
        )

    st.dataframe(frame_data, use_container_width=True)

    # Grad-CAM visualizations
    if any(fp["gradcam_available"] for fp in results["frame_predictions"]):
        st.subheader("🔍 Explainable AI - Grad-CAM Visualizations")
        st.info(
            "Red areas indicate regions the AI model focused on when making its decision."
        )

        # Display Grad-CAM images in columns
        gradcam_frames = [
            fp for fp in results["frame_predictions"] if fp["gradcam_available"]
        ]

        cols = st.columns(min(3, len(gradcam_frames)))
        for i, frame_pred in enumerate(gradcam_frames[:6]):  # Show max 6 images
            col_idx = i % len(cols)
            with cols[col_idx]:
                try:
                    gradcam_url = f"{BACKEND_URL}/api/v1/gradcam/{results['video_id']}/{frame_pred['frame_number']}"
                    st.image(
                        gradcam_url,
                        caption=f"Frame {frame_pred['frame_number']} - {frame_pred['prediction']} ({frame_pred['confidence']:.1%})",
                        use_column_width=True,
                    )
                except Exception as e:
                    st.error(
                        f"Could not load Grad-CAM for frame {frame_pred['frame_number']}"
                    )

    # Technical details
    with st.expander("🔧 Technical Details"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Model Information:**")
            st.write(f"- Architecture: {results['metadata']['model_architecture']}")
            st.write(f"- Version: {results['model_version']}")
            st.write(
                f"- Confidence Threshold: {results['metadata']['confidence_threshold']}"
            )

        with col2:
            st.write("**Video Information:**")
            st.write(f"- Original Filename: {results['metadata']['original_filename']}")
            st.write(
                f"- File Size: {results['metadata']['file_size'] / (1024 * 1024):.1f} MB"
            )
            st.write(f"- Duration: {results['metadata']['video_duration']:.1f} seconds")
            st.write(f"- Frames Extracted: {results['metadata']['frames_extracted']}")


def main():
    """Main application function"""

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🎬 Deepfake Video Detector</h1>
        <p>AI-powered video authenticity analysis with explainable AI</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Check backend status
    if not check_backend_health():
        st.error(
            "🚨 Backend service is not available. Please ensure the backend is running."
        )
        st.info("To start the backend, run: `python run_backend.py`")
        return

    # Sidebar configuration
    st.sidebar.header("⚙️ Analysis Configuration")

    num_frames = st.sidebar.slider(
        "Number of frames to analyze",
        min_value=1,
        max_value=20,
        value=5,
        help="More frames provide better accuracy but take longer to process",
    )

    enable_gradcam = st.sidebar.checkbox(
        "Enable Grad-CAM visualizations",
        value=True,
        help="Generate explainable AI visualizations (increases processing time)",
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Threshold for classifying as deepfake",
    )

    # Information section
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Information")
    st.sidebar.info(
        "This tool uses advanced AI to detect deepfake videos. "
        "Upload a video file and get detailed analysis with explanations."
    )

    st.sidebar.markdown("**Supported formats:**")
    for fmt in SUPPORTED_FORMATS:
        st.sidebar.write(f"• {fmt.upper()}")

    st.sidebar.markdown(f"**Max file size:** {MAX_FILE_SIZE // (1024 * 1024)} MB")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📁 Upload Video")

        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=[fmt[1:] for fmt in SUPPORTED_FORMATS],  # Remove dots
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
        )

        if uploaded_file is not None:
            # File validation
            file_size = len(uploaded_file.getvalue())
            file_extension = Path(uploaded_file.name).suffix.lower()

            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.write(f"📊 Size: {file_size / (1024 * 1024):.1f} MB")
            st.write(f"📄 Format: {file_extension.upper()}")

            # Validation checks
            if file_size > MAX_FILE_SIZE:
                st.error(
                    f"❌ File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)} MB"
                )
                return

            if file_extension not in SUPPORTED_FORMATS:
                st.error(
                    f"❌ Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
                )
                return

            # Video preview
            st.video(uploaded_file)

            # Analysis button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                # Reset file pointer
                uploaded_file.seek(0)

                # Perform analysis
                results = upload_and_analyze_video(
                    uploaded_file, num_frames, enable_gradcam, confidence_threshold
                )

                if results:
                    st.session_state["analysis_results"] = results

    with col2:
        st.header("📈 Quick Stats")

        # Get model info
        try:
            model_info = requests.get(f"{BACKEND_URL}/api/v1/models/info").json()
            st.metric("Model", model_info["architecture"])
            st.metric("Status", "🟢 Ready" if model_info["loaded"] else "🔴 Not Ready")
        except:
            st.metric("Backend", "🔴 Offline")

        # Usage tips
        st.markdown("---")
        st.header("💡 Tips")
        st.markdown("""
        - **Better accuracy:** Use more frames (5-10)
        - **Faster processing:** Use fewer frames (1-3)
        - **Best quality:** Upload high-resolution videos
        - **Grad-CAM:** Shows what the AI focuses on
        """)

    # Display results if available
    if "analysis_results" in st.session_state:
        st.markdown("---")
        st.header("📋 Analysis Results")
        display_results(st.session_state["analysis_results"])

        # Clear results button
        if st.button("🗑️ Clear Results"):
            del st.session_state["analysis_results"]
            st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🔬 <strong>Educational Tool</strong> - From Hasif's Workspace</p>
        <p>⚠️ For research and demonstration purposes</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

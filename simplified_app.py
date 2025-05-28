"""
Simplified Single-File Deepfake Detector Application
Streamlit app with embedded model for easy deployment
From Hasif's Workspace
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import tempfile
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Deepfake Video Detector", page_icon="🎬", layout="wide")


# Simple model definition (embedded)
class SimpleDeepfakeDetector(nn.Module):
    """Simplified deepfake detector for demonstration"""

    def __init__(self, num_classes=1):
        super(SimpleDeepfakeDetector, self).__init__()

        # Use a lightweight backbone
        import torchvision.models as models

        self.backbone = models.mobilenet_v2(pretrained=True)

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


@st.cache_resource
def load_model():
    """Load the deepfake detection model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleDeepfakeDetector(num_classes=1)
        model.to(device)
        model.eval()

        # Try to load trained weights if available
        model_path = "data/models/deepfake_detector_best.pth"
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                st.success("✅ Trained model loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Could not load trained weights: {e}")
                st.info("Using pre-trained MobileNetV2 weights")
        else:
            st.info(
                "ℹ️ Using pre-trained MobileNetV2 weights (no fine-tuned model found)"
            )

        return model, device

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


def preprocess_frame(frame):
    """Preprocess frame for model input"""
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(frame)


def extract_frames(video_path, num_frames=5):
    """Extract frames from video"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []

        # Calculate frame indices
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        current_frame = 0

        while cap.isOpened() and len(frames) < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in frame_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            current_frame += 1

        cap.release()
        return frames

    except Exception as e:
        st.error(f"Error extracting frames: {e}")
        return []


def predict_frames(model, device, frames):
    """Predict on extracted frames"""
    if not frames:
        return []

    predictions = []

    with torch.no_grad():
        for i, frame in enumerate(frames):
            try:
                # Preprocess frame
                input_tensor = preprocess_frame(frame).unsqueeze(0).to(device)

                # Get prediction
                output = model(input_tensor)
                probability = torch.sigmoid(output).item()

                prediction = "Deepfake" if probability > 0.5 else "Real"
                confidence = (
                    probability if prediction == "Deepfake" else 1 - probability
                )

                predictions.append(
                    {
                        "frame_number": i + 1,
                        "prediction": prediction,
                        "confidence": confidence,
                        "probability": probability,
                    }
                )

            except Exception as e:
                st.error(f"Error predicting frame {i + 1}: {e}")
                predictions.append(
                    {
                        "frame_number": i + 1,
                        "prediction": "Error",
                        "confidence": 0.0,
                        "probability": 0.0,
                    }
                )

    return predictions


def main():
    """Main application function"""

    # Header
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h1>🎬 Deepfake Video Detector</h1>
        <p>Simple AI-powered video authenticity analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load model
    model, device = load_model()

    if model is None:
        st.error("❌ Could not load model. Please check your setup.")
        return

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    num_frames = st.sidebar.slider("Frames to analyze", 1, 10, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**ℹ️ About**")
    st.sidebar.info(
        "This tool analyzes videos to detect potential deepfakes using AI. "
        "Upload a video file and get instant results!"
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📁 Upload Video")

        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV",
        )

        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size:.1f} MB)")

            # Show video
            st.video(uploaded_file)

            # Analyze button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name

                try:
                    # Extract frames
                    with st.spinner("Extracting frames..."):
                        frames = extract_frames(temp_path, num_frames)

                    if not frames:
                        st.error("❌ Could not extract frames from video")
                        return

                    st.success(f"✅ Extracted {len(frames)} frames")

                    # Predict
                    with st.spinner("Analyzing frames..."):
                        predictions = predict_frames(model, device, frames)

                    # Calculate overall result
                    if predictions:
                        avg_prob = np.mean([p["probability"] for p in predictions])
                        overall_prediction = "Deepfake" if avg_prob > 0.5 else "Real"
                        overall_confidence = (
                            avg_prob
                            if overall_prediction == "Deepfake"
                            else 1 - avg_prob
                        )

                        # Display results
                        st.markdown("---")
                        st.header("📊 Results")

                        # Overall result
                        if overall_prediction == "Real":
                            st.success(
                                f"✅ **{overall_prediction}** (Confidence: {overall_confidence:.1%})"
                            )
                        else:
                            st.error(
                                f"⚠️ **{overall_prediction}** (Confidence: {overall_confidence:.1%})"
                            )

                        # Frame details
                        st.subheader("Frame Analysis")

                        for pred in predictions:
                            col_frame, col_pred, col_conf = st.columns([1, 1, 1])

                            with col_frame:
                                st.write(f"Frame {pred['frame_number']}")

                            with col_pred:
                                if pred["prediction"] == "Real":
                                    st.success(pred["prediction"])
                                else:
                                    st.error(pred["prediction"])

                            with col_conf:
                                st.write(f"{pred['confidence']:.1%}")

                        # Show sample frames
                        st.subheader("Sample Frames")
                        frame_cols = st.columns(min(3, len(frames)))

                        for i, frame in enumerate(frames[:3]):
                            with frame_cols[i]:
                                st.image(
                                    frame,
                                    caption=f"Frame {i + 1}: {predictions[i]['prediction']}",
                                    use_column_width=True,
                                )

                finally:
                    # Cleanup
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

    with col2:
        st.header("📈 Model Info")

        device_info = "🟢 GPU" if device.type == "cuda" else "🔵 CPU"
        st.metric("Device", device_info)
        st.metric("Architecture", "MobileNetV2")
        st.metric("Status", "🟢 Ready")

        st.markdown("---")
        st.header("💡 Tips")
        st.markdown("""
        - **Upload quality videos** for better accuracy
        - **More frames** = better analysis but slower
        - **Shorter videos** process faster
        - Results are for **demonstration purposes**
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🔬 <strong>Educational Tool</strong> - From Hasif's Workspace</p>
        <p>⚠️ For learning and demonstration purposes</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

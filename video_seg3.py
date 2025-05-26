import streamlit as st
import torch
import os
import tempfile
import cv2
import numpy as np
import logging

from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_model():
    try:
        model_path = "yolov8n-seg.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        # Load YOLOv8 model with full checkpoint (not weights_only)
        model = torch.load(model_path, map_location="cpu", weights_only=False)

        # Sometimes the loaded dict needs to be rewrapped (older checkpoints)
        if isinstance(model, dict) and "model" in model:
            from ultralytics.nn.tasks import attempt_load_one_weight
            model = attempt_load_one_weight(model_path)

        model.to("cpu")
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None


def process_frame(frame, model):
    try:
        results = model(frame)
        return results[0].plot()
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame


def process_video(video_file, model):
    try:
        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        input_temp.write(video_file.read())
        input_temp.close()

        cap = cv2.VideoCapture(input_temp.name)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = process_frame(frame, model)
            out.write(annotated_frame)

        cap.release()
        out.release()
        os.unlink(input_temp.name)

        return output_temp.name
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        st.error(f"Failed to process video: {str(e)}")
        return None


def main():
    st.title("🎥 YOLOv8 Segmentation App")
    st.write("Upload a video or image to perform object segmentation using YOLOv8.")

    uploaded_file = st.file_uploader("📤 Upload a video or image", type=["mp4", "avi", "jpg", "png"])

    if uploaded_file is not None:
        with st.spinner("Loading model..."):
            model = load_model()
        if model is None:
            st.stop()

        file_type = uploaded_file.type
        if "video" in file_type:
            st.video(uploaded_file)
            if st.button("🚀 Process Video"):
                with st.spinner("Processing video..."):
                    output_path = process_video(uploaded_file, model)
                    if output_path:
                        st.success("✅ Video processed!")
                        st.video(output_path)
                        with open(output_path, "rb") as f:
                            st.download_button("⬇️ Download", f, file_name="segmented_video.mp4")
                        os.unlink(output_path)
        elif "image" in file_type:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, channels="BGR", caption="Uploaded Image")
            if st.button("🧠 Process Image"):
                with st.spinner("Segmenting image..."):
                    annotated_image = process_frame(image, model)
                    st.image(annotated_image, channels="BGR", caption="Segmented Image")
                    _, buffer = cv2.imencode(".png", annotated_image)
                    st.download_button("⬇️ Download", data=buffer.tobytes(), file_name="segmented_image.png",
                                       mime="image/png")
        else:
            st.error("Unsupported file type.")


if __name__ == "__main__":
    main()

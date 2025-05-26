import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import logging
import os
import tempfile
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import SegmentationModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv

# Allowlist required classes for torch.load to work with weights_only=True
add_safe_globals([SegmentationModel, Sequential, Conv])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_model():
    """
    Load the YOLOv8 segmentation model with CPU compatibility.
    """
    try:
        model_path = "yolov8n-seg.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        model = YOLO(model_path)
        model.to("cpu")
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None


def process_frame(frame, model):
    """
    Process a single frame with YOLOv8 segmentation.
    Returns the frame with segmentation masks overlaid.
    """
    try:
        results = model(frame)
        annotated_frame = results[0].plot()
        return annotated_frame
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame


def process_video(video_file, model):
    """
    Process a video file and return the path to the segmented video.
    """
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
        logger.info("Video processing completed")

        return output_temp.name
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        st.error(f"Failed to process video: {str(e)}")
        return None
    finally:
        if os.path.exists(input_temp.name):
            os.unlink(input_temp.name)


def main():
    st.title("Video Segmentation App with YOLOv8")
    st.write("Upload a video or image to perform object segmentation using YOLOv8.")

    uploaded_file = st.file_uploader("Choose a video or image", type=["mp4", "avi", "jpg", "png"])

    if uploaded_file is not None:
        with st.spinner("Loading model..."):
            model = load_model()
        if model is None:
            st.stop()

        file_type = uploaded_file.type

        if "video" in file_type:
            st.video(uploaded_file)
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    output_path = process_video(uploaded_file, model)
                    if output_path:
                        st.success("Video processed successfully!")
                        st.video(output_path)
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Segmented Video",
                                data=f,
                                file_name="segmented_video.mp4",
                                mime="video/mp4"
                            )
                        os.unlink(output_path)
        elif "image" in file_type:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, channels="BGR", caption="Uploaded Image")
            if st.button("Process Image"):
                with st.spinner("Processing image..."):
                    annotated_image = process_frame(image, model)
                    st.image(annotated_image, channels="BGR", caption="Segmented Image")
                    _, buffer = cv2.imencode(".png", annotated_image)
                    st.download_button(
                        label="Download Segmented Image",
                        data=buffer.tobytes(),
                        file_name="segmented_image.png",
                        mime="image/png"
                    )
        else:
            st.error("Unsupported file type")


if __name__ == "__main__":
    main()

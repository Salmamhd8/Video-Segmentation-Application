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

# Allowlist required classes to fix weights_only=True error
add_safe_globals([SegmentationModel, Sequential, Conv])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
@st.cache_resource
def load_model():
    """
    Load the YOLOv8 segmentation model with CPU compatibility.
    """
    try:
        model_path = "yolov8n-seg.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        # Use torch.hub to avoid weights_only issue
        model = torch.hub.load('ultralytics/ultralytics', 'custom', path=model_path, trust_repo=True)
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
        # Run inference
        results = model(frame)
        # Get annotated frame with masks
        annotated_frame = results[0].plot()  # Plot masks and boxes
        return annotated_frame
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame  # Return original frame if processing fails


def process_video(video_file, model):
    """
    Process a video file and return the path to the segmented video.
    """
    try:
        # Create temporary files
        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        # Save uploaded video to temporary file
        input_temp.write(video_file.read())
        input_temp.close()

        # Read input video
        cap = cv2.VideoCapture(input_temp.name)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Process frame
            annotated_frame = process_frame(frame, model)
            # Write to output video
            out.write(annotated_frame)

        # Release resources
        cap.release()
        out.release()
        logger.info("Video processing completed")

        # Return output video path
        return output_temp.name
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        st.error(f"Failed to process video: {str(e)}")
        return None
    finally:
        # Clean up input temp file
        if os.path.exists(input_temp.name):
            os.unlink(input_temp.name)


def main():
    """
    Main Streamlit app function.
    """
    st.title("Video Segmentation App with YOLOv8")
    st.write("Upload a video or image to perform object segmentation using YOLOv8.")

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Choose a video or image", type=["mp4", "avi", "jpg", "png"])

    if uploaded_file is not None:
        # Check file type
        file_type = uploaded_file.type
        if "video" in file_type:
            st.video(uploaded_file)
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    output_path = process_video(uploaded_file, model)
                    if output_path:
                        st.success("Video processed successfully!")
                        st.video(output_path)
                        # Provide download button
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Segmented Video",
                                data=f,
                                file_name="segmented_video.mp4",
                                mime="video/mp4"
                            )
                        # Clean up output file
                        os.unlink(output_path)
        elif "image" in file_type:
            # Read and display image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, channels="BGR", caption="Uploaded Image")
            if st.button("Process Image"):
                with st.spinner("Processing image..."):
                    annotated_image = process_frame(image, model)
                    st.image(annotated_image, channels="BGR", caption="Segmented Image")
                    # Convert to bytes for download
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
import os
import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import SegmentationModel, DetectionModel
from torch.nn.modules.container import Sequential

# Patch pour la désérialisation PyTorch (YOLOv8 custom)
add_safe_globals([SegmentationModel, DetectionModel, Sequential])

# Configuration de la page Streamlit
st.set_page_config(layout="wide")
st.title("🎥 Advanced Video Segmentation Application")
st.write("Upload a video to extract an object and change its background")

# Chargement du modèle YOLOv8 localement
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "yolov8n-seg.pt")
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLOv8 model: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# Effet cartoon pour l'arrière-plan
def apply_cartoon_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
    color = cv2.bilateralFilter(frame, 5, 150, 150)
    return cv2.bitwise_and(color, color, mask=edges)

# Classes COCO courantes
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    16: "dog", 17: "cat", 18: "horse", 19: "sheep", 20: "cow"
}

# Interface utilisateur
uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        selected_class = st.selectbox("Object to segment", list(COCO_CLASSES.values()))
        class_id = [k for k, v in COCO_CLASSES.items() if v == selected_class][0]
    with col2:
        bg_option = st.radio("Background type", ["Color", "Blur", "Transparent", "Cartoon", "Custom image"])

    if bg_option == "Color":
        bg_color = st.color_picker("Choose background color", "#00FF00")
        bg_value = np.array([int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    elif bg_option == "Custom image":
        bg_image = st.file_uploader("Upload background image", type=["jpg", "jpeg", "png"])
        if bg_image:
            bg_img = Image.open(bg_image)
            st.image(bg_img, caption="Background image", width=200)

    if st.button("Process Video"):
        if bg_option == "Custom image" and not bg_image:
            st.warning("Please upload a background image")
        else:
            with st.spinner("Processing... Please wait"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_file.read())
                    temp_path = tfile.name

                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    st.error("Could not open video file")
                    os.unlink(temp_path)
                    st.stop()

                width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
                width, height = width - (width % 2), height - (height % 2)

                output_path = "output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                if bg_option == "Custom image" and bg_image:
                    try:
                        custom_bg = np.array(bg_img.convert('RGB'))
                        custom_bg = cv2.resize(custom_bg, (width, height))
                    except Exception as e:
                        st.error(f"Error processing background image: {str(e)}")
                        os.unlink(temp_path)
                        st.stop()

                progress_bar = st.progress(0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processed_frames = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    try:
                        results = model(frame, classes=[class_id], conf=0.5)
                    except Exception as e:
                        st.error(f"Error in segmentation: {str(e)}")
                        break

                    if len(results[0]) > 0:
                        try:
                            mask = results[0].masks[0].data[0].cpu().numpy() * 255
                            mask = cv2.resize(mask, (width, height))

                            if bg_option == "Color":
                                background = np.full_like(frame, bg_value)
                            elif bg_option == "Blur":
                                background = cv2.blur(frame, (50, 50))
                            elif bg_option == "Cartoon":
                                background = apply_cartoon_effect(frame)
                            elif bg_option == "Custom image":
                                background = custom_bg.copy()
                            else:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                                frame[:, :, 3] = mask.astype(np.uint8)
                                background = np.zeros_like(frame)

                            if bg_option != "Transparent":
                                result = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8)) + \
                                         cv2.bitwise_and(background, background, mask=255 - mask.astype(np.uint8))
                            else:
                                result = frame
                        except Exception as e:
                            st.error(f"Error processing frame: {str(e)}")
                            result = frame
                    else:
                        result = frame

                    out.write(result)
                    processed_frames += 1
                    progress_bar.progress(processed_frames / frame_count)

                cap.release()
                out.release()
                os.unlink(temp_path)

            st.success("Processing complete! You can now download the video.")
            with open(output_path, "rb") as f:
                st.download_button("Download processed video", f, file_name="segmented_video.mp4", mime="video/mp4")

            if os.path.exists(output_path):
                os.unlink(output_path)

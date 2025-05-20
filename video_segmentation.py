import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# Configuration de la page
st.set_page_config(layout="wide")
st.title("🎥 Application de Segmentation Vidéo Avancée")

st.write("Téléchargez une vidéo pour extraire un objet et changer son fond")

# Chargement du modèle
@st.cache_resource
def load_model():
    model = YOLO("yolov8n-seg.pt")  # Modèle de segmentation
    return model

model = load_model()

# Fonction pour l'effet cartoon
def apply_cartoon_effect(frame):
    """Application de l'effet cartoon"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    edges = cv2.adaptiveThreshold(blur, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 5, 5)
    color = cv2.bilateralFilter(frame, 5, 150, 150)
    return cv2.bitwise_and(color, color, mask=edges)

# Dictionnaire des classes COCO
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    16: "dog", 17: "cat", 18: "horse", 19: "sheep", 20: "cow"
}

# Interface utilisateur
uploaded_file = st.file_uploader("Choisissez une vidéo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Paramètres
    col1, col2 = st.columns(2)
    with col1:
        selected_class = st.selectbox("Objet à segmenter", list(COCO_CLASSES.values()))
        class_id = [k for k, v in COCO_CLASSES.items() if v == selected_class][0]
    with col2:
        bg_option = st.radio("Type de fond", ["Couleur", "Flou", "Transparent", "Cartoon", "Image personnalisée"])

    if bg_option == "Couleur":
        bg_color = st.color_picker("Choisir une couleur de fond", "#00FF00")
        bg_value = np.array([int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    elif bg_option == "Image personnalisée":
        bg_image = st.file_uploader("Télécharger une image de fond", type=["jpg", "jpeg", "png"])
        if bg_image:
            bg_img = Image.open(bg_image)
            st.image(bg_img, caption="Image de fond", width=200)

    # Traitement
    if st.button("Traiter la vidéo"):
        if bg_option == "Image personnalisée" and not bg_image:
            st.warning("Veuillez télécharger une image de fond")
        else:
            with st.spinner("Traitement en cours... Veuillez patienter"):
                # Sauvegarde temporaire
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_file.read())
                    temp_path = tfile.name

                # Lecture vidéo
                cap = cv2.VideoCapture(temp_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

                # Ajustement des dimensions
                width, height = width - (width % 2), height - (height % 2)

                # Préparation output
                output_path = "resultat.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # Préparation fond personnalisé si image
                if bg_option == "Image personnalisée" and bg_image:
                    custom_bg = np.array(bg_img.convert('RGB'))
                    custom_bg = cv2.resize(custom_bg, (width, height))

                # Traitement frame par frame
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Segmentation
                    results = model(frame, classes=[class_id], conf=0.5)

                    if len(results[0]) > 0:  # Si objet détecté
                        mask = results[0].masks[0].data[0].cpu().numpy() * 255
                        mask = cv2.resize(mask, (width, height))

                        # Préparation fond
                        if bg_option == "Couleur":
                            background = np.full_like(frame, bg_value)
                        elif bg_option == "Flou":
                            background = cv2.blur(frame, (50, 50))
                        elif bg_option == "Cartoon":
                            background = apply_cartoon_effect(frame)
                        elif bg_option == "Image personnalisée":
                            background = custom_bg.copy()
                        else:  # Transparent
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                            frame[:, :, 3] = mask.astype(np.uint8)
                            background = np.zeros_like(frame)

                        # Combinaison
                        if bg_option != "Transparent":
                            result = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8)) + \
                                     cv2.bitwise_and(background, background, mask=255 - mask.astype(np.uint8))
                        else:
                            result = frame
                    else:
                        result = frame  # Si aucun objet détecté

                    out.write(result)

                # Nettoyage
                cap.release()
                out.release()
                os.unlink(temp_path)

            # Affichage résultat sans la vidéo
            st.success("Traitement terminé ! Vous pouvez maintenant télécharger la vidéo.")

            # Option de téléchargement
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Télécharger la vidéo traitée",
                    data=f,
                    file_name="video_segmente.mp4",
                    mime="video/mp4"
                )
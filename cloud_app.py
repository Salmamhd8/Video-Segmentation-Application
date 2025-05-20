import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import requests

# Configuration de la page
st.set_page_config(layout="wide")
st.title("🎥 Application de Segmentation Vidéo Avancée")

st.write("Téléchargez une vidéo pour extraire un objet et changer son fond")


# Téléchargement du modèle si absent (pour Streamlit Cloud)
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n-seg.pt")  # Essaye de charger le modèle local
    except:
        st.warning("Téléchargement du modèle YOLOv8n-seg... (cela peut prendre quelques minutes)")
        model = YOLO('yolov8n-seg.pt')  # Téléchargement automatique
    model.to('cpu')  # Force l'utilisation du CPU
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


# Dictionnaire des classes COCO (version simplifiée)
COCO_CLASSES = {
    0: "person", 2: "car", 3: "motorcycle",
    16: "dog", 17: "cat"
}

# Interface utilisateur
uploaded_file = st.file_uploader("Choisissez une vidéo (max 50MB)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Vérification de la taille du fichier
    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
        st.error("La vidéo est trop volumineuse (max 50MB)")
        st.stop()

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
            with st.spinner("Traitement en cours... (cela peut prendre du temps)"):
                # Sauvegarde temporaire
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_file.read())
                    temp_path = tfile.name

                # Lecture vidéo
                cap = cv2.VideoCapture(temp_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

                # Ajustement des dimensions pour le codec
                width, height = width - (width % 2), height - (height % 2)

                # Préparation output
                output_path = "resultat.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # Préparation fond personnalisé si image
                if bg_option == "Image personnalisée" and bg_image:
                    custom_bg = np.array(bg_img.convert('RGB'))
                    custom_bg = cv2.resize(custom_bg, (width, height))

                # Barre de progression
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processed_frames = 0

                # Traitement frame par frame
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Segmentation (sur CPU)
                    results = model(frame, classes=[class_id], conf=0.5, device='cpu')

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
                    processed_frames += 1
                    progress_bar.progress(processed_frames / total_frames)

                # Nettoyage
                cap.release()
                out.release()
                os.unlink(temp_path)

            # Affichage résultat
            st.success("Traitement terminé !")

            # Prévisualisation du résultat
            st.video(output_path)

            # Option de téléchargement
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Télécharger la vidéo traitée",
                    data=f,
                    file_name="video_segmente.mp4",
                    mime="video/mp4"
                )

            # Nettoyage du fichier temporaire
            try:
                os.remove(output_path)
            except:
                pass
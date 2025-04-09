import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

def load_model():
    """Charge le modèle YOLOv8 pour la détection d'objet"""
    model = YOLO('yolov8n.pt')  # Modèle de détection
    return model

def process_image_detection(image, model, conf_threshold):
    """Traite une image avec détection YOLO"""
    # Faire la prédiction
    results = model(image, conf=conf_threshold)
    
    # Obtenir le premier résultat
    result = results[0]
    
    # Créer une copie de l'image originale
    output_image = image.copy()
    
    # Dictionnaire pour compter les classes détectées
    detected_classes = {}
    
    # Calculer la taille appropriée du texte en fonction de la taille de l'image
    image_height = image.shape[0]
    base_font_scale = 0.8  # Taille de base du texte
    font_scale = max(1.0, (image_height / 1080) * base_font_scale)  # Ajustement selon la résolution
    font_thickness = max(2, int(font_scale * 2))  # Épaisseur proportionnelle à la taille
    box_thickness = max(2, int(font_scale * 2))  # Épaisseur des boîtes
    
    # Pour chaque détection
    for box in result.boxes:
        # Obtenir les coordonnées de la boîte
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Obtenir la classe et la confiance
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = model.names[class_id]
        
        # Ajouter à notre compteur
        if class_name in detected_classes:
            detected_classes[class_name] += 1
        else:
            detected_classes[class_name] = 1
        
        # Choisir une couleur pour cette classe
        color = (hash(class_name) % 256, (hash(class_name) * 17) % 256, (hash(class_name) * 43) % 256)
        
        # Dessiner la boîte avec une épaisseur adaptative
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, box_thickness)
        
        # Préparer le texte
        text = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculer la taille du texte pour un meilleur placement
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Créer un fond pour le texte avec un padding
        padding = int(font_scale * 5)
        cv2.rectangle(output_image, 
                     (x1, y1 - text_height - padding), 
                     (x1 + text_width + padding, y1),
                     color, 
                     -1)  # -1 remplit le rectangle
        
        # Ajouter le texte
        cv2.putText(output_image, 
                    text, 
                    (x1 + padding//2, y1 - padding//2),
                    font,
                    font_scale,
                    (255, 255, 255),  # Texte en blanc
                    font_thickness)
    
    return output_image, detected_classes

def process_video(video_path, output_path, model, conf_threshold):
    """Traite une vidéo avec détection YOLO"""
    cap = cv2.VideoCapture(video_path)
    
    # Obtenir les propriétés de la vidéo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Créer l'objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Barre de progression
    progress_bar = st.progress(0)
    frame_text = st.empty()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convertir BGR à RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Traiter l'image
        processed_frame, _ = process_image_detection(frame_rgb, model, conf_threshold)
        
        # Convertir RGB à BGR pour la sauvegarde
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Écrire la frame
        out.write(processed_frame_bgr)
        
        # Mettre à jour la progression
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        frame_text.text(f"Traitement de l'image {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path

def main():
    st.title("Application de Détection d'Objets")
    
    # Chargement du modèle
    with st.spinner("Chargement du modèle YOLOv8 pour la détection d'objet..."):
        @st.cache_resource
        def get_model():
            return load_model()
        
        model = get_model()
    
    # Configuration de la sidebar
    st.sidebar.title("Configuration")
    
    # Type de fichier
    file_type = st.sidebar.radio(
        "Type de fichier",
        ["Image", "Vidéo"]
    )
    
    # Paramètres de détection
    conf_threshold = st.sidebar.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    
    if file_type == "Image":
        # Upload d'image
        uploaded_file = st.file_uploader("Choisissez une image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Lire l'image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Afficher l'image originale
            st.subheader("Image originale")
            st.image(image_rgb)
            
            # Traiter l'image
            processed_image, detected_objects = process_image_detection(image_rgb, model, conf_threshold)
            
            # Afficher l'image traitée
            st.subheader("Image traitée")
            st.image(processed_image)
            
            # Afficher les statistiques
            st.sidebar.subheader("Objets détectés")
            for obj, count in detected_objects.items():
                st.sidebar.text(f"{obj}: {count}")
            
            # Bouton de téléchargement
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            if is_success:
                btn = st.download_button(
                    label="Télécharger l'image traitée",
                    data=buffer.tobytes(),
                    file_name="image_traitée.png",
                    mime="image/png"
                )
    
    else:  # Vidéo
        # Upload de vidéo
        uploaded_file = st.file_uploader("Choisissez une vidéo", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Créer un fichier temporaire pour la vidéo
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            # Créer un fichier temporaire pour la sortie
            output_path = os.path.join(tempfile.gettempdir(), "video_traitée.mp4")
            
            # Traiter la vidéo
            st.subheader("Traitement de la vidéo")
            processed_video_path = process_video(tfile.name, output_path, model, conf_threshold)
            
            # Nettoyer le fichier temporaire d'entrée
            tfile.close()
            os.unlink(tfile.name)
            
            # Lire le fichier de sortie
            with open(processed_video_path, 'rb') as f:
                video_bytes = f.read()
            
            # Bouton de téléchargement
            st.download_button(
                label="Télécharger la vidéo traitée",
                data=video_bytes,
                file_name="video_traitée.mp4",
                mime="video/mp4"
            )
            
            # Nettoyer le fichier temporaire de sortie
            os.unlink(processed_video_path)
            
            # Afficher la vidéo traitée
            st.subheader("Vidéo traitée")
            st.video(video_bytes)

if __name__ == "__main__":
    main()
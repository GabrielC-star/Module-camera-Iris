import cv2
import numpy as np
from picamera2 import Picamera2
from module_camera_PC.config import SEUIL_CONFIANCE, TAILLE_VISAGE, TAILLE_MIN_VISAGE, FICHIER_MODELE
from module_camera_PC.dataset import charger_dataset
from module_camera_PC.detection_salle import detecter_salle, afficher_salle, ANALYSER_TOUTES_LES_N_FRAMES


def reconnaitre_visages():
    _, _, noms = charger_dataset()
    modele = cv2.face.LBPHFaceRecognizer_create()
    modele.read(FICHIER_MODELE)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # --- Initialisation Picamera2 ---
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 360)},
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
    )
    picam2.configure(config)
    picam2.start()

    compteur_frames = 0

    print("Reconnaissance en cours — Appuie sur 'q' pour quitter")

    try:
        while True:
            # --- Capture Picamera2 + conversion RGB → BGR ---
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=TAILLE_MIN_VISAGE
            )

            # --- Reconnaissance faciale ---
            for (x, y, w, h) in faces:
                visage = cv2.resize(gray[y:y+h, x:x+w], TAILLE_VISAGE)
                label_id, confiance = modele.predict(visage)

                if confiance < SEUIL_CONFIANCE:
                    nom = noms[label_id]
                    couleur = (0, 255, 0)
                    texte = f"{nom} ({confiance:.1f})"
                else:
                    nom = "Inconnu"
                    couleur = (0, 0, 255)
                    texte = f"Inconnu ({confiance:.1f})"

                cv2.rectangle(frame, (x, y), (x+w, y+h), couleur, 2)
                cv2.putText(frame, texte, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)

            # --- Détection de salle toutes les N frames ---
            if compteur_frames % ANALYSER_TOUTES_LES_N_FRAMES == 0:
                detecter_salle(frame)

            compteur_frames += 1

            # --- Affichage de la salle ---
            afficher_salle(frame)

            cv2.imshow("IRIS - Reconnaissance", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
import cv2
import numpy as np
import os
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Variable globale pour stocker le numéro de salle
salle_detectee = ""

# ------------------------------------------------------------------ #
#                        RECONNAISSANCE FACIALE                       #
# ------------------------------------------------------------------ #

def charger_dataset(dossier="dataset"):
    visages = []
    labels = []
    noms = {}

    for label_id, nom in enumerate(os.listdir(dossier)):
        noms[label_id] = nom
        chemin_personne = os.path.join(dossier, nom)

        for fichier in os.listdir(chemin_personne):
            chemin_img = os.path.join(chemin_personne, fichier)
            img = cv2.imread(chemin_img, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                visages.append(img)
                labels.append(label_id)

    return visages, labels, noms


def entrainer_modele():
    print("Chargement du dataset...")
    visages, labels, noms = charger_dataset()

    print(f"  {len(visages)} images chargées pour {len(noms)} personnes")

    modele = cv2.face.LBPHFaceRecognizer_create()
    modele.train(visages, np.array(labels))
    modele.save("modele_iris.yml")

    print("Modèle entraîné et sauvegardé !")
    return modele, noms


# ------------------------------------------------------------------ #
#                        DÉTECTION DE SALLE                           #
# ------------------------------------------------------------------ #

def detecter_salle(frame):
    global salle_detectee

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    texte_brut = pytesseract.image_to_string(gray, config=config)

    pattern = r'[A-Z]\d{3}'
    correspondances = re.findall(pattern, texte_brut.upper())

    if correspondances:
        salle_detectee = correspondances[0]
        print(f"Salle détectée : {salle_detectee}")

    return salle_detectee


# ------------------------------------------------------------------ #
#                         BOUCLE PRINCIPALE                           #
# ------------------------------------------------------------------ #

def reconnaitre_visages():
    global salle_detectee

    _, _, noms = charger_dataset()
    modele = cv2.face.LBPHFaceRecognizer_create()
    modele.read("modele_iris.yml")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    SEUIL_CONFIANCE = 80
    ANALYSER_TOUTES_LES_N_FRAMES = 15
    compteur_frames = 0

    print("Reconnaissance en cours — Appuie sur 'q' pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        # --- Reconnaissance faciale ---
        for (x, y, w, h) in faces:
            visage = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
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

        # --- Affichage permanent de la salle en mémoire ---
        if salle_detectee:
            cv2.putText(frame, f"Salle : {salle_detectee}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 165, 0), 2)

        cv2.imshow("IRIS - Reconnaissance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDernière salle détectée : '{salle_detectee}'")
    return salle_detectee


# --- MAIN ---
entrainer_modele()
salle = reconnaitre_visages()
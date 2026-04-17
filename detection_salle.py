import cv2
import pytesseract
import re
from config import ANALYSER_TOUTES_LES_N_FRAMES

salle_detectee = ""


def detecter_salle(frame):
    global salle_detectee

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    texte_brut = pytesseract.image_to_string(gray, config=config)

    if texte_brut.strip():
        print(f"Tesseract lit : '{texte_brut.strip()}'")

    pattern = r'[A-Z]\d{3}'
    correspondances = re.findall(pattern, texte_brut.upper())

    if correspondances:
        salle_detectee = correspondances[0]
        print(f"Salle détectée : {salle_detectee}")

    return salle_detectee


def afficher_salle(frame):
    """Affiche le numéro de salle en mémoire sur la frame"""
    if salle_detectee:
        cv2.putText(frame, f"Salle : {salle_detectee}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 165, 0), 2)
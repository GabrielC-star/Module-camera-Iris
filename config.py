import pytesseract

# --- Tesseract ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Reconnaissance faciale ---
SEUIL_CONFIANCE = 80
TAILLE_VISAGE = (100, 100)
TAILLE_MIN_VISAGE = (80, 80)

# --- Détection de salle ---
ANALYSER_TOUTES_LES_N_FRAMES = 15

# --- Fichiers ---
DOSSIER_DATASET = "dataset"
FICHIER_MODELE = "modele_iris.yml"
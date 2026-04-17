# 📷 Module Caméra — IRIS

> Intelligent Robot for Interactive Services — EPF Engineering School

Module de reconnaissance faciale et détection de numéros de salle en temps réel, développé pour le robot IRIS.

---

## 📁 Structure du projet

```
Module_camera/
├── main.py                     # Point d'entrée
├── config.py                   # Paramètres globaux
├── dataset.py                  # Chargement des images et entraînement
├── detection_salle.py          # Détection OCR des numéros de salle
├── reconnaissance_faciale.py   # Boucle caméra et reconnaissance
├── capture_visage.py           # Script de capture des photos
└── dataset/                    # Dossier des photos (créé automatiquement)
    └── [Nom]/
        ├── 0.jpg
        ├── 1.jpg
        └── ...
```

---

## ⚙️ Prérequis

### Python
- Python 3.8 ou supérieur

### Tesseract OCR

**Windows :**
1. Télécharger l'installeur : https://github.com/UB-Mannheim/tesseract/wiki
2. Choisir `tesseract-ocr-w64-setup-5.x.x.exe` (64 bits)
3. Installer (chemin par défaut : `C:\Program Files\Tesseract-OCR`)

**Linux / Raspberry Pi :**
```bash
sudo apt install tesseract-ocr
```

---

## 📦 Installation des dépendances Python

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement virtuel
# Windows :
venv\Scripts\activate
# Linux / Mac :
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Lancement

### Étape 1 — Capturer les photos d'un utilisateur

```bash
python enregistrement_visage.py
```

- Une fenêtre s'ouvre avec le flux de la caméra
- Placer le visage dans le rectangle vert
- Appuyer sur **ESPACE** pour prendre une photo
- Répéter **30 fois** en variant légèrement les angles
- Appuyer sur **Q** pour quitter

> Les photos sont sauvegardées automatiquement dans `dataset/[Nom]/`

### Étape 2 — Lancer la reconnaissance

```bash
python main.py
```

Cela va :
1. Entraîner le modèle sur les photos capturées
2. Sauvegarder le modèle dans `modele_iris.yml`
3. Lancer la reconnaissance faciale et la détection de salle en temps réel

---

## 🎮 Contrôles

| Touche | Action |
|--------|--------|
| `Q` | Quitter le programme |
| `ESPACE` | Prendre une photo (mode capture uniquement) |

---

## 🖥️ Affichage

| Couleur | Signification |
|---------|---------------|
| 🟢 Vert | Visage reconnu |
| 🔴 Rouge | Visage inconnu |
| 🟠 Orange | Numéro de salle détecté |

---

## 🔧 Configuration

Tous les paramètres sont centralisés dans `config.py` :

```python
# Chemin vers Tesseract (Windows uniquement)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Seuil de confiance pour la reconnaissance faciale
# Plus il est bas = plus strict / Plus il est haut = plus permissif
SEUIL_CONFIANCE = 80

# Fréquence d'analyse OCR (toutes les N frames)
ANALYSER_TOUTES_LES_N_FRAMES = 15
```

---

## 📸 Format des numéros de salle

Le module détecte les numéros au format **une lettre + 3 chiffres** :

```
✅ P902   ✅ C106   ✅ B207
❌ 902    ❌ PP902  ❌ P90
```

Pour de meilleurs résultats avec l'OCR :
- Texte imprimé, police claire, taille minimum 2-3 cm
- Fond blanc, texte noir (contraste maximal)
- Caméra à 20-40 cm du texte
- Bonne luminosité, sans reflets

---

## 🐛 Problèmes courants

| Erreur | Cause | Solution |
|--------|-------|----------|
| `TesseractNotFoundError` | Tesseract non trouvé | Vérifier le chemin dans `config.py` |
| `cv2.face` introuvable | Mauvais package OpenCV | Installer `opencv-contrib-python` |
| Visage toujours "Inconnu" | Seuil trop bas | Augmenter `SEUIL_CONFIANCE` à 90 dans `config.py` |
| Aucune photo capturée | Visage non détecté | Améliorer l'éclairage |
| Salle jamais détectée | Texte trop petit/flou | Rapprocher la caméra du texte |

---

## 📋 Dépendances

| Bibliothèque | Version | Usage |
|---|---|---|
| `opencv-python` | ≥ 4.5 | Flux vidéo et détection de visages |
| `opencv-contrib-python` | ≥ 4.5 | Modèle LBPH (reconnaissance faciale) |
| `numpy` | ≥ 1.21 | Traitement des arrays d'images |
| `pytesseract` | ≥ 0.3 | Interface Python pour Tesseract OCR |
| `Tesseract OCR` | ≥ 5.0 | Moteur OCR pour la détection de salle |

---

## 👥 Équipe

Projet réalisé à l'**EPF École d'Ingénieurs** dans le cadre du projet IRIS.
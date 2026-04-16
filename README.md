# 🧠 Projet IRIS – Reconnaissance Faciale avec OpenCV

## 📌 Description

Ce projet permet de réaliser un système de reconnaissance faciale en plusieurs étapes :

- 📸 Capture d’images de visages via webcam
- 🗂️ Création d’un dataset par utilisateur
- 🏋️ Entraînement d’un modèle LBPH (OpenCV)
- 🎥 Reconnaissance faciale en temps réel

---

## 🚀 Installation

### 1. Cloner le projet
git clone https://github.com/GabrielC-star/Module-camera-Iris.git
cd Module_camera

---

### 2. Créer un environnement virtuel
python -m venv venv

---

### 3. Activer le venv

Git Bash :
source venv/Scripts/activate

---

### 4. Installer les dépendances
pip install -r requirements.txt

---

## 📸 Étape 1 – Capture des visages

python Enregistrement_visage.py  -> pas à lancer si vous avez déja un dataset de photo

- La webcam s’ouvre
- Appuie sur ESPACE pour capturer une image
- Les images sont enregistrées dans data/NOM_UTILISATEUR/

---

## 🏋️ Étape 2 – Entraînement du modèle

python Entrainement_reconnaissance.py

- Charge les images du dataset
- Entraîne un modèle LBPH
- Génère trainer.yml

⚠️ Installer obligatoire :
pip install opencv-contrib-python

---

## 🎥 Étape 3 – Reconnaissance en temps réel

python reconnaissance.py

- Ouvre la webcam
- Détecte les visages
- Affiche le nom reconnu en direct

---

## 📁 Structure du projet

Module_camera/
│
├── venv/                              # Environnement virtuel (non versionné)
├── data/                              # Dataset des visages
│   ├── Gabriel/
│   ├── Autres_utilisateurs/
│
├── Enregistrement_visage.py          # Capture des images webcam
├── Entrainement_reconnaissance.py    # Entraînement du modèle LBPH
├── reconnaissance.py                 # Reconnaissance en temps réel
│
├── trainer.yml                      # Modèle entraîné (généré)
├── requirements.txt
└── README.md

---

## ⚙️ Technologies utilisées

- Python 3.10+
- OpenCV (opencv-contrib-python obligatoire)
- NumPy

---

## 🚫 Bonnes pratiques

- Ne jamais versionner venv/
- Toujours activer le venv avant d’exécuter
- Utiliser un seul environnement Python
- Utiliser requirements.txt pour partager les dépendances

---

## 🧠 Principe du projet

1. Capture d’images
2. Prétraitement
3. Entraînement LBPH
4. Reconnaissance en temps réel

---

## 👤 Auteur

Projet étudiant – EPF
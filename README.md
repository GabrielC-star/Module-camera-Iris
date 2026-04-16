# 🎯 Reconnaissance faciale – Projet IRIS

Ce module permet de capturer des visages via une caméra et d’entraîner un modèle de reconnaissance faciale.

---

## 📦 Installation

Installer les dépendances nécessaires :

```bash
pip install opencv-python opencv-contrib-python numpy
```

---

## 📷 Source vidéo

* Actuellement : webcam du PC (`index 0`)
* À terme : migration vers **Picamera2** pour Raspberry Pi

---

## 🚀 Utilisation

### 1. Capture des visages

Lancer le script :

```bash
python Enregistrement_visage.py
```

👉 Ce script :

* capture environ **30 images** du visage
* enregistre les données pour l’entraînement

💡 **Conseil :**
Augmenter le nombre de photos améliore la précision du modèle.

---

### 2. Entraînement du modèle

Après la capture des images, lancer le script d’entraînement.

---

## 🧠 Reconnaissance faciale

Le système utilise un seuil de confiance pour déterminer si un visage est reconnu.

```python
SEUIL_CONFIANCE = 80
```

### Interprétation :

* **Confiance < seuil** → visage reconnu ✅
* **Confiance > seuil** → visage inconnu ❌

---

## ⚙️ Améliorations possibles

* Augmenter le nombre d’images d’entraînement
* Améliorer les conditions de capture (lumière, angles)
* Implémenter Picamera2 pour Raspberry Pi
* Ajouter une interface utilisateur

---

## 📁 Structure du projet (exemple)

```
.
├── Enregistrement_visage.py
├── dataset/
├── Entrainement&reconnaissance.py/
└── README.md
```

---

## 🎓 Contexte

Projet réalisé dans le cadre du projet **IRIS** – robot autonome pour campus étudiant.

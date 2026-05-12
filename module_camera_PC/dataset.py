import cv2
import numpy as np
import os
from config import DOSSIER_DATASET, FICHIER_MODELE, TAILLE_VISAGE


def charger_dataset(dossier=DOSSIER_DATASET):
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
    modele.save(FICHIER_MODELE)

    print("Modèle entraîné et sauvegardé !")
    return modele, noms
import cv2
import numpy as np
import os

def charger_dataset(dossier="dataset"):
    """Charge toutes les images et leurs labels"""
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
    """Entraîne le modèle LBPH sur le dataset"""
    print("Chargement du dataset...")
    visages, labels, noms = charger_dataset()
    
    print(f"  {len(visages)} images chargées pour {len(noms)} personnes")
    
    # Modèle LBPH — léger et efficace pour Raspberry Pi
    modele = cv2.face.LBPHFaceRecognizer_create()
    modele.train(visages, np.array(labels))
    modele.save("modele_iris.yml")
    
    print("Modèle entraîné et sauvegardé !")
    return modele, noms


def reconnaitre_visages():
    """Lance la reconnaissance en temps réel"""
    
    # Chargement du modèle
    _, _, noms = charger_dataset()
    modele = cv2.face.LBPHFaceRecognizer_create()
    modele.read("modele_iris.yml")
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    cap = cv2.VideoCapture(0)
    
    # Seuil de confiance — en dessous : reconnu, au dessus : inconnu
    SEUIL_CONFIANCE = 80
    
    print("Reconnaissance en cours — Appuie sur 'q' pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        for (x, y, w, h) in faces:
            visage = gray[y:y+h, x:x+w]
            visage = cv2.resize(visage, (100, 100))
            
            # Prédiction
            label_id, confiance = modele.predict(visage)
            
            if confiance < SEUIL_CONFIANCE:
                nom = noms[label_id]
                couleur = (0, 255, 0)  # Vert = reconnu
                texte = f"{nom} ({confiance:.1f})"
            else:
                nom = "Inconnu"
                couleur = (0, 0, 255)  # Rouge = inconnu
                texte = f"Inconnu ({confiance:.1f})"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), couleur, 2)
            cv2.putText(frame, texte, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
        
        cv2.imshow("IRIS - Reconnaissance faciale", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# --- MAIN ---
# Lance d'abord l'entraînement, puis la reconnaissance
entrainer_modele()
reconnaitre_visages()
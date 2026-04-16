import cv2
import numpy as np
import os

NOM_UTILISATEUR = "Gabriel"
SEUIL_CONFIANCE = 80

def entrainer():
    dossier = f"dataset/{NOM_UTILISATEUR}"
    visages = []
    labels = []
    
    for fichier in os.listdir(dossier):
        img = cv2.imread(f"{dossier}/{fichier}", cv2.IMREAD_GRAYSCALE)
        if img is not None:
            visages.append(img)
            labels.append(0)  # Un seul label car un seul utilisateur
    
    print(f"{len(visages)} images chargées pour {NOM_UTILISATEUR}")
    
    modele = cv2.face.LBPHFaceRecognizer_create()
    modele.train(visages, np.array(labels))
    modele.save("modele_iris.yml")
    print("Modèle entraîné et sauvegardé !")
    
    return modele


def reconnaitre(modele):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    
    print("Reconnaissance en cours — Appuie sur 'q' pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        for (x, y, w, h) in faces:
            visage = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            _, confiance = modele.predict(visage)
            
            if confiance < SEUIL_CONFIANCE:
                texte = f"{NOM_UTILISATEUR} ({confiance:.1f})"
                couleur = (0, 255, 0)   # Vert = reconnu
            else:
                texte = f"Inconnu ({confiance:.1f})"
                couleur = (0, 0, 255)   # Rouge = inconnu
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), couleur, 2)
            cv2.putText(frame, texte, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
        
        cv2.imshow("IRIS - Reconnaissance", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# --- MAIN ---
modele = entrainer()
reconnaitre(modele)
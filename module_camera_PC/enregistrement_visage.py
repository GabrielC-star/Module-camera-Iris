import cv2
import os

def capture_visage(nom, nb_photos=30):
    dossier = f"dataset/{nom}"
    os.makedirs(dossier, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    compteur = 0
    print(f"Capture pour {nom} — Appuie sur ESPACE pour prendre une photo")
    
    while compteur < nb_photos:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Photos: {compteur}/{nb_photos}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f"Capture - {nom}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and len(faces) > 0:
            x, y, w, h = faces[0]
            visage = gray[y:y+h, x:x+w]
            visage = cv2.resize(visage, (100, 100))
            cv2.imwrite(f"{dossier}/{compteur}.jpg", visage)
            compteur += 1
            print(f"  Photo {compteur}/{nb_photos} sauvegardée")
        
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Capture terminée pour {nom} !")

# Un seul utilisateur
capture_visage("Gabriel")
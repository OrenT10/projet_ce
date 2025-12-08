""" Avant de lancer le script il faut:
    - Se connecter au réseau sur lequel se trouve le logiciel
    - Configurer l'IP manuellement avec le panneau de configuration
    - IP : 192.168.1.26 """


import cv2
import mediapipe as mp
import os
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import socket

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Imports locaux
from pressure_manager import LivePressureVisualizer
from visual_engine import PressureRing, FootTracker

# --- CONFIGURATION FICHIER ---
MODEL_FILE = 'pose_landmarker_full.task'
WINDOW_NAME = "Foot Pressure Tracker"

def ask_user_mode():
    """
    Affiche une fenêtre contextuelle pour choisir le mode.
    Retourne True si on veut la SIMULATION, False si on veut le LIVE.
    """
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True)
    
    # Question inversée comme demandé :
    # OUI = LIVE
    # NON = SIMULATION
    want_live_mode = messagebox.askyesno(
        title="Configuration du Démarrage",
        message=(
            "Voulez-vous lancer le mode LIVE (Écoute UDP) ?\n\n"
            "✅ OUI = Mode LIVE (Port 5005)\n"
            "❌ NON = Mode SIMULATION (Données aléatoires)"
        ),
        icon='question'
    )
    
    root.destroy()
    
    # Si l'utilisateur veut le LIVE (True), alors SIMULATION_MODE doit être False
    return not want_live_mode

def ask_user_mass():
    """
    Demande la masse de l'utilisateur pour calibrer MAX_FORCE_N.
    Retourne la masse (float) ou une valeur par défaut.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Demande la masse
    mass = simpledialog.askfloat(
        "Calibration Utilisateur", 
        "Entrez la masse de l'utilisateur (kg) :",
        minvalue=10.0, 
        maxvalue=200.0
    )
    
    root.destroy()
    
    if mass is None:
        print("Annulation ou erreur: masse par défaut 70kg utilisée.")
        return 70.0
    return mass


def main():
    # 1. Choix du mode
    print("⏳ En attente du choix utilisateur...")
    SIMULATION_MODE = ask_user_mode()
    
    # 2. Calibration de la Force
    user_mass = ask_user_mass()

    # Calcul dynamique de MAX_FORCE_N
    calculated_max_force = (user_mass*9.81 / 2.0) * 0.7
    
    print(f"--- CALIBRATION ---")
    print(f"Masse utilisateur : {user_mass} kg")
    print(f"MAX_FORCE_N calculé : {calculated_max_force:.2f}")
    
    # Instanciation avec le paramètre dynamique
    pressure_vis = LivePressureVisualizer(
        simulation_mode=SIMULATION_MODE, 
        max_force=calculated_max_force
    )
    
    mode_str = "SIMULATION" if SIMULATION_MODE else "LIVE UDP"
    print(f"🔹 Mode sélectionné : {mode_str}")

    # 2. Initialisation Data
    visualizer = LivePressureVisualizer(simulation_mode=SIMULATION_MODE)
    visualizer.start_listener()

    # 3. Initialisation MediaPipe
    path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    if not os.path.exists(path):
        print(f"❌ Erreur: Modèle '{MODEL_FILE}' introuvable.")
        return

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # 4. Initialisation Caméra
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("⚠️ Caméra 1 échec. Tentative Caméra 0...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Aucune caméra trouvée.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # --- CONFIGURATION FENÊTRE PLEIN ÉCRAN ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 5. Initialisation Objets Visuels
    drawer = PressureRing()
    left_foot = FootTracker(is_right=False, drawer=drawer)
    right_foot = FootTracker(is_right=True, drawer=drawer)

    print("✅ Système prêt. En attente de données...")
    while not visualizer.update_state():
        time.sleep(0.1)
    print("🚀 Lancement de la visualisation !")

    # 6. Variable d'enregistrement vidéo
    is_recording = False
    video_writer = None
    REC_COLOR = (0, 0, 255)

    # 7. Boucle Principale
    try:
        while cap.isOpened():
            # A. Data Pression
            visualizer.update_state()
            r_L, r_R = visualizer.get_radii()
            c_L, c_R = visualizer.get_zone_colors()
            left_foot.update_pressure(r_L, c_L)
            right_foot.update_pressure(r_R, c_R)

            # B. Vision
            success, frame = cap.read()
            if not success: break
            
            # !! IMPORTANT : Définir h, w ici pour qu'ils soient toujours disponibles
            h, w = frame.shape[:2] 
            clean_bg = frame.copy()
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = detector.detect_for_video(mp_img, ts)

            # C. Mise à jour Position
            if result.pose_landmarks:
                for pose in result.pose_landmarks:
                    # h, w sont déjà définis plus haut
                    left_foot.update_landmarks(pose, w, h)
                    right_foot.update_landmarks(pose, w, h)

            # D. Dessin
            left_foot.render(frame, clean_bg)
            right_foot.render(frame, clean_bg)

            # Indicateur de mode discret
            cv2.putText(frame, f"MODE: {mode_str}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # --- LOGIQUE D'ENREGISTREMENT MODIFIÉE ---
            
            # 2. Écrire l'image si l'enregistrement est actif
            if is_recording and video_writer:
                video_writer.write(frame)
                
                # Ajout d'un indicateur visuel "REC"
                cv2.putText(frame, "REC", (w - 100, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, REC_COLOR, 2)


            # 3. Gestion des touches
            key = cv2.waitKey(5) & 0xFF
            
            if key == 27: # ESC pour quitter
                break
            
            elif key == ord('r'): # 'r' pour Démarrer/Arrêter l'enregistrement
                if not is_recording:
                    # Démarrer l'enregistrement
                    print("Enregistrement DÉMARRÉ")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    # Utiliser les dimensions (w, h) définies au début de la boucle
                    video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))
                    is_recording = True
                    REC_COLOR = (0, 0, 255) # Rouge
                else:
                    # Arrêter l'enregistrement
                    print("Enregistrement ARRÊTÉ. Fichier 'output.avi' sauvegardé.")
                    is_recording = False
                    if video_writer:
                        video_writer.release()
                    video_writer = None
            
            # Clignotement de l'indicateur REC (optionnel mais sympa)
            if is_recording:
                if (ts // 500) % 2 == 0: # Clignote toutes les 500ms
                    REC_COLOR = (0, 0, 255)
                else:
                    REC_COLOR = (0, 0, 150)


            # (La ligne d'enregistrement originale a été supprimée)
            # cv2.VideoWriter('output.avi', ...).write(frame)  <- SUPPRIMÉE

            cv2.imshow(WINDOW_NAME, frame)

    finally:
        cap.release()
        
        # 4. Nettoyage final
        # Assurez-vous de fermer le fichier vidéo si l'utilisateur quitte
        # pendant l'enregistrement (sinon le fichier sera corrompu)
        if video_writer:
            print("Nettoyage : fermeture du fichier vidéo.")
            video_writer.release()
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
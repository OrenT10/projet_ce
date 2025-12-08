import cv2
import mediapipe as mp
import time
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

# Imports locaux (inchangés)
from data_manager import DataManager
from visual_engine import Visualizer
from stability_analyzer import StabilityAnalyzer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_FILE = 'pose_landmarker_full.task' 

def ask_configuration():
    # ... (inchangé) ...
    root = tk.Tk()
    root.withdraw()
    is_live = messagebox.askyesno("Mode", "OUI = LIVE (UDP)\nNON = SIMULATION")
    simulation = not is_live
    mass = simpledialog.askfloat("Masse", "Masse utilisateur (kg) :", minvalue=10, maxvalue=200)
    root.destroy()
    return simulation, (mass if mass else 70.0)

def main():
    sim_mode, user_mass = ask_configuration()
    max_force = (user_mass * 9.81 / 2.0) * 0.9 
    
    data_mgr = DataManager(simulation_mode=sim_mode)
    visualizer = Visualizer(max_force=max_force)
    analyzer = StabilityAnalyzer(buffer_size=50)
    
    if not os.path.exists(MODEL_FILE):
        print(f"ERREUR: {MODEL_FILE} manquant.")
        return

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # --- CONFIGURATION VIDÉO ---
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps_output = 20.0 
    out = cv2.VideoWriter('output_session.mp4', fourcc, fps_output, (actual_width, actual_height))
    # ---------------------------

    data_mgr.start()
    
    # Variable d'état pour l'enregistrement
    is_recording = False  # <--- AJOUT : On commence sans enregistrer
    
    print(">>> 'r' = ENREGISTRER / STOP || 'ESC' = QUITTER")
    
    last_valid_sensor_data = None
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # 1. Vision
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            ts_ms = int(time.time() * 1000)
            detection_result = detector.detect_for_video(mp_image, ts_ms)
            
            # 2. Données Capteurs
            new_data = data_mgr.get_latest_data()
            if new_data:
                last_valid_sensor_data = new_data
                analyzer.update(new_data)
            
            # 3. Rendu Graphique
            if last_valid_sensor_data:
                score, status = analyzer.get_stability_info()
                visualizer.process_and_draw(frame, last_valid_sensor_data, detection_result)
                
                # Jauge
                color = (0, 255, 0)
                if score < 75: color = (0, 165, 255)
                if score < 40: color = (0, 0, 255)
                
                cv2.rectangle(frame, (50, 50), (350, 80), (50, 50, 50), -1)
                bar_width = int(3.0 * score)
                cv2.rectangle(frame, (50, 50), (50 + bar_width, 80), color, -1)
                cv2.putText(frame, f"{status} ({int(score)}%)", (50, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "Attente donnees...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # --- GESTION ENREGISTREMENT ---
            if is_recording:
                # 1. On dessine un point rouge "REC" en haut à droite
                cv2.circle(frame, (actual_width - 50, 50), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (actual_width - 100, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 2. On écrit l'image dans le fichier
                out.write(frame)
            else:
                # Optionnel : Afficher "PAUSE" (en gris) pour être sûr
                cv2.putText(frame, "PAUSE", (actual_width - 120, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            # ------------------------------

            cv2.imshow('Foot Pressure Tracker', frame)
            
            # --- GESTION CLAVIER AMÉLIORÉE ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27: # Touche ESC
                break
            elif key == ord('r'): # Touche 'r' (minuscule)
                is_recording = not is_recording # On inverse l'état
                state_msg = "ON" if is_recording else "OFF"
                print(f">>> Enregistrement : {state_msg}")
            # ---------------------------------
                
    finally:
        data_mgr.stop()
        cap.release()
        if 'out' in locals():
            out.release()
            print(">>> Vidéo sauvegardée.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
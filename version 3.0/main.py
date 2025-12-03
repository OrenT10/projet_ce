import cv2
import mediapipe as mp
import time
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

# Imports locaux
from data_manager import DataManager
from visual_engine import Visualizer
from stability_analyzer import StabilityAnalyzer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_FILE = 'pose_landmarker_full.task' 

def ask_configuration():
    root = tk.Tk()
    root.withdraw()
    is_live = messagebox.askyesno("Mode", "OUI = LIVE (UDP)\nNON = SIMULATION")
    simulation = not is_live
    mass = simpledialog.askfloat("Masse", "Masse utilisateur (kg) :", minvalue=10, maxvalue=200)
    root.destroy()
    return simulation, (mass if mass else 70.0)

def main():
    sim_mode, user_mass = ask_configuration()
    # Facteur de sécurité augmenté (0.9) pour éviter saturation trop rapide
    max_force = (user_mass * 9.81 / 2.0) * 0.9 
    
    data_mgr = DataManager(simulation_mode=sim_mode)
    visualizer = Visualizer(max_force=max_force)
    analyzer = StabilityAnalyzer(buffer_size=50) # 1 sec à 50Hz
    
    # Init Vision
    if not os.path.exists(MODEL_FILE):
        print(f"ERREUR: {MODEL_FILE} manquant.")
        return

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # Choix Caméra
    cap = cv2.VideoCapture(1)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    data_mgr.start()
    
    print(">>> Lancement. 'ESC' pour quitter.")
    
    # Variable pour la persistance
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
            
            # 2. Données Capteurs (Gestion Persistance)
            new_data = data_mgr.get_latest_data()
            
            if new_data:
                # Si nouvelle donnée, on met à jour la référence et l'analyse
                last_valid_sensor_data = new_data
                analyzer.update(new_data)
            
            # 3. Rendu (On utilise last_valid_sensor_data qu'il soit nouveau ou vieux)
            if last_valid_sensor_data:
                # Récup info stabilité
                score, status = analyzer.get_stability_info()
                
                # Dessin Pieds/Pression
                visualizer.process_and_draw(frame, last_valid_sensor_data, detection_result)
                
                # Dessin Jauge
                color = (0, 255, 0) # Vert
                if score < 75: color = (0, 165, 255) # Orange
                if score < 40: color = (0, 0, 255) # Rouge
                
                cv2.rectangle(frame, (50, 50), (350, 80), (50, 50, 50), -1)
                bar_width = int(3.0 * score)
                cv2.rectangle(frame, (50, 50), (50 + bar_width, 80), color, -1)
                cv2.putText(frame, f"{status} ({int(score)}%)", (50, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "Attente donnees...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow('Foot Pressure Tracker', frame)
            if cv2.waitKey(1) & 0xFF == 27: break
                
    finally:
        data_mgr.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
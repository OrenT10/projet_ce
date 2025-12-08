import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Imports requis pour la classe LivePressureVisualizer ---
import socket
import re
import threading
import queue
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# -----------------------------------------------------------

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

############################################################################
#
# --- CLASSE LIVEPRESSUREVISUALIZER (définie dans le fichier précédent) ---
#
############################################################################

class LivePressureVisualizer:
    """
    Gère la réception, le traitement et la préparation des données de 
    pression OpenGo en temps réel pour la visualisation.
    """

    # --- Configuration UDP ---
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    BUFFER_SIZE = 2048
    QUEUE_MAX = 1000

    # --- Configuration Visuelle ---
    SENSORS_FORE_EXT = [8, 11, 12, 13, 16]
    SENSORS_FORE_INT = [7, 9, 10, 14, 15]
    SENSORS_HIND_EXT = [2, 4, 6]
    SENSORS_HIND_INT = [1, 3, 5]

    MAX_FORCE_N = 500.0
    BASE_RADIUS = 0.2    # Rayon normalisé min
    MAX_RADIUS = 1.0     # Rayon normalisé max
    RING_WIDTH = 0.1     # Note: non utilisé par le dessin OpenCV
    CMAP_NAME = 'YlOrBr'

    def __init__(self):
        self.data_queue = queue.Queue(maxsize=self.QUEUE_MAX)
        self.t0 = None
        self.listener_thread = None
        
        self.last_data_packet = None
        self.current_time = 0.0
        self.radius_L = self.BASE_RADIUS
        self.radius_R = self.BASE_RADIUS
        
        self.zones_L = self._get_empty_zones()
        self.zones_R = self._get_empty_zones()
        
        self.max_pressure = 1.0
        self.norm = mcolors.Normalize(vmin=0, vmax=self.max_pressure)
        self.cmap = cm.get_cmap(self.CMAP_NAME)

    def _get_empty_zones(self):
        return {'Fore_Int': 0, 'Fore_Ext': 0, 'Hind_Ext': 0, 'Hind_Int': 0}

    @staticmethod
    def _parse_opengo_udp(raw_data: str):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_data)
        if len(nums) < 51:
            return None
        v = list(map(float, nums[:51]))
        return {
            "time_raw": v[0],
            "left": {"pressure": v[7:23], "total_force": v[25]},
            "right": {"pressure": v[32:48], "total_force": v[50]},
        }

    @staticmethod
    def _aggregate_zones(pressures):
        return {
            'Fore_Ext': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_FORE_EXT),
            'Fore_Int': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_FORE_INT),
            'Hind_Ext': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_HIND_EXT),
            'Hind_Int': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_HIND_INT),
        }

    def _udp_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.UDP_IP, self.UDP_PORT))
        print(f"🔌 [LiveVisualizer] Écoute UDP sur {self.UDP_IP}:{self.UDP_PORT} ...")

        while True:
            data, _ = sock.recvfrom(self.BUFFER_SIZE)
            text = data.decode("utf-8", errors="ignore").strip()
            parsed = self._parse_opengo_udp(text)
            
            if not parsed:
                continue
                
            if self.t0 is None:
                self.t0 = parsed["time_raw"]
                
            parsed["time"] = round(parsed["time_raw"] - self.t0, 3)
            del parsed["time_raw"]
            
            try:
                self.data_queue.put_nowait(parsed)
            except queue.Full:
                pass

    def _calculate_radius(self, force):
        scaled_force = min(force, self.MAX_FORCE_N) / self.MAX_FORCE_N
        return self.BASE_RADIUS + (scaled_force * (self.MAX_RADIUS - self.BASE_RADIUS))

    def start_listener(self):
        if self.listener_thread is None:
            self.listener_thread = threading.Thread(target=self._udp_listener, daemon=True)
            self.listener_thread.start()
            print("[LiveVisualizer] Thread d'écoute démarré.")

    def update_state(self):
        new_data = None
        try:
            while True:
                new_data = self.data_queue.get_nowait()
        except queue.Empty:
            pass

        if new_data is None and self.last_data_packet is None:
            return False
        
        if new_data is not None:
            self.last_data_packet = new_data
        
        data = self.last_data_packet
        
        self.current_time = data["time"]
        self.radius_L = self._calculate_radius(data["left"]["total_force"])
        self.radius_R = self._calculate_radius(data["right"]["total_force"])
        
        self.zones_L = self._aggregate_zones(data["left"]["pressure"])
        self.zones_R = self._aggregate_zones(data["right"]["pressure"])
        
        max_L = max(self.zones_L.values()) if self.zones_L else 0
        max_R = max(self.zones_R.values()) if self.zones_R else 0
        self.max_pressure = max(max_L, max_R, 1.0)
        self.norm = mcolors.Normalize(vmin=0, vmax=self.max_pressure)
        
        return True

    # --- GETTERS PUBLICS ---
    def get_radii(self):
        """Retourne (rayon_gauche_norm, rayon_droit_norm) [0.2 à 1.0]"""
        return (self.radius_L, self.radius_R)

    def get_zone_colors(self):
        """Retourne les couleurs RGBA (0-1) pour chaque quadrant."""
        colors_L = {
            'Fore_Int': self.cmap(self.norm(self.zones_L['Fore_Int'])),
            'Fore_Ext': self.cmap(self.norm(self.zones_L['Fore_Ext'])),
            'Hind_Ext': self.cmap(self.norm(self.zones_L['Hind_Ext'])),
            'Hind_Int': self.cmap(self.norm(self.zones_L['Hind_Int'])),
        }
        colors_R = {
            'Fore_Int': self.cmap(self.norm(self.zones_R['Fore_Int'])),
            'Fore_Ext': self.cmap(self.norm(self.zones_R['Fore_Ext'])),
            'Hind_Ext': self.cmap(self.norm(self.zones_R['Hind_Ext'])),
            'Hind_Int': self.cmap(self.norm(self.zones_R['Hind_Int'])),
        }
        return (colors_L, colors_R)


############################################################################
#
# --- SCRIPT DE TRACKING VIDÉO (Modifié) ---
#
############################################################################

# --- 1. Initialisation du Visualiseur de Pression ---
visualizer = LivePressureVisualizer()
visualizer.start_listener()

# --- 2. Configuration du modèle MediaPipe ---
model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker_full.task')
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Fichier '{model_path}' non trouvé.\n"
        "Baixe em: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,  # Détecte une personne    
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.PoseLandmarker.create_from_options(options)

# --- 3. Paramètres de visualisation (OpenCV) ---
BASE_RADIUS_PX = 30    # Rayon minimum (en pixels)
MAX_RADIUS_PX = 80     # Rayon maximum (en pixels)
# MAX_FORCE_SIM (supprimé, car nous utilisons MAX_FORCE_N de la classe)

# --- 4. Nouvelles Fonctions de Dessin (Utilisant la classe) ---

def rgba_to_bgr(rgba):
    """Convertit Matplotlib RGBA (0-1) en OpenCV BGR (0-255)"""
    r, g, b, a = rgba
    return (int(b * 255), int(g * 255), int(r * 255))

def draw_foot_pressure(image, cx, cy, normalized_radius, colors_rgba, is_right=False):
    """
    Dessine les 4 secteurs de pression en utilisant les données pré-calculées
    par LivePressureVisualizer.
    """
    
    # Convertir le rayon normalisé (0.2-1.0) en rayon en pixels (30-80px)
    # (On utilise les constantes de la classe pour la plage normalisée)
    norm_range = visualizer.MAX_RADIUS - visualizer.BASE_RADIUS
    pixel_range = MAX_RADIUS_PX - BASE_RADIUS_PX
    
    radius_ratio = (normalized_radius - visualizer.BASE_RADIUS) / norm_range
    radius_px = int(BASE_RADIUS_PX + (radius_ratio * pixel_range))

    if radius_px <= 0:
        return # Ne rien dessiner si le rayon est nul

    # Obtenir les couleurs BGR
    c_fore_int = rgba_to_bgr(colors_rgba['Fore_Int'])
    c_fore_ext = rgba_to_bgr(colors_rgba['Fore_Ext'])
    c_hind_ext = rgba_to_bgr(colors_rgba['Hind_Ext'])
    c_hind_int = rgba_to_bgr(colors_rgba['Hind_Int'])

    # Définition des zones (angles en degrés pour cv2.ellipse)
    # Mapping CORRECT (basé sur la classe visualizer)
    if not is_right:
        # Pied Gauche
        # (Couleur, AngleDébut, AngleFin)
        zones = [
            (c_fore_int, 0, 90),     # Avant-Interne (Top-Right)
            (c_fore_ext, 90, 180),   # Avant-Externe (Top-Left)
            (c_hind_ext, 180, 270),  # Arrière-Externe (Bottom-Left)
            (c_hind_int, 270, 360),  # Arrière-Interne (Bottom-Right)
        ]
    else:
        # Pied Droit (Miroir horizontal du pied gauche)
        zones = [
            (c_fore_int, 90, 180),   # Avant-Interne (devient Top-Left)
            (c_fore_ext, 0, 90),     # Avant-Externe (devient Top-Right)
            (c_hind_ext, 270, 360),  # Arrière-Externe (devient Bottom-Right)
            (c_hind_int, 180, 270),  # Arrière-Interne (devient Bottom-Left)
        ]

    # Dessiner les 4 secteurs (wedges)
    for color, start_angle, end_angle in zones:
        cv2.ellipse(
            image,
            (cx, cy),
            (radius_px, radius_px),
            0,
            start_angle,
            end_angle,
            color,
            thickness=-1  # Rempli
        )
        # Dessiner une fine bordure noire pour séparer les secteurs
        cv2.ellipse(
            image,
            (cx, cy),
            (radius_px, radius_px),
            0,
            start_angle,
            end_angle,
            (0, 0, 0),
            thickness=1
        )

# --- 5. Initialisation de la webcam ---
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_timestamp_ms = 0

print("En attente des premières données UDP...")
# Attend d'avoir reçu au moins un paquet de données avant de démarrer
while not visualizer.update_state():
    time.sleep(0.1)
print("Données reçues. Démarrage de la capture vidéo.")

# --- 6. Boucle Principale ---
try:
    while cap.isOpened():
        
        # --- A. Mise à jour des données de pression ---
        # (Récupère le paquet le plus récent dans la file)
        visualizer.update_state()
        
        # Obtenir les données de pression calculées pour CETTE frame
        radius_L_norm, radius_R_norm = visualizer.get_radii()
        colors_L_rgba, colors_R_rgba = visualizer.get_zone_colors()
        
        # --- B. Capture et Traitement Vidéo ---
        success, image = cap.read()
        if not success:
            continue

        # Processa a imagem com MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Incrémenter le timestamp pour le mode VIDEO de MediaPipe
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        annotated_image = image.copy()

        # --- C. Dessin de la Superposition ---
        
        # Processa cada pessoa detectada (até 2)
        for pose_landmarks in detection_result.pose_landmarks:
            h, w = image.shape[:2]

            if len(pose_landmarks) < 33:
                continue

            try:
                # Indices: 29=LEFT_HEEL, 31=LEFT_FOOT_INDEX, 30=RIGHT_HEEL, 32=RIGHT_FOOT_INDEX
                left_heel = pose_landmarks[29]
                left_toe = pose_landmarks[31]
                right_heel = pose_landmarks[30]
                right_toe = pose_landmarks[32]
            except IndexError:
                continue

            # Centre des pieds (coordonnées en pixels)
            left_center = (
                int((left_heel.x + left_toe.x) * w / 2),
                int((left_heel.y + left_toe.y) * h / 2)
            )
            right_center = (
                int((right_heel.x + right_toe.x) * w / 2),
                int((right_heel.y + right_toe.y) * h / 2)
            )

            # --- Remplacement de la simulation ---
            # (Plus besoin de simuler, on utilise les données réelles)

            # Dessiner le pied GAUCHE
            draw_foot_pressure(
                annotated_image,
                left_center[0] - 10  , left_center[1] - 10, # Décalage pour mieux observer
                radius_L_norm,
                colors_L_rgba,
                is_right=False
            )
            
            # Dessiner le pied DROIT
            draw_foot_pressure(
                annotated_image,
                right_center[0] - 10, right_center[1] - 10, # Décalage pour mieux observer
                radius_R_norm,
                colors_R_rgba,
                is_right=True
            )

        # Afficher le résultat
        cv2.imshow('Pression des pieds - Live UDP + MediaPipe', annotated_image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Capture arrêtée.")
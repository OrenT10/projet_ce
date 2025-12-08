import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math
import random  # Pour la simulation

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
# --- CLASSE LIVEPRESSUREVISUALIZER (Avec Mode Simulation) ---
#
############################################################################

class LivePressureVisualizer:
    """
    Gère la réception (ou simulation) des données de pression.
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
    BASE_RADIUS = 0.2
    MAX_RADIUS = 1.0
    CMAP_NAME = 'YlOrBr' # Jaune -> Orange -> Brun

    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
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

    # --- Simulation ---
    def _simulation_loop(self):
        """Génère des données aléatoires fluides pour le test."""
        print("🔮 [LiveVisualizer] Mode SIMULATION activé.")
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            # Simuler une marche : force sinusoïdale
            # Pied gauche et droit en opposition de phase
            force_L = (math.sin(elapsed * 3) + 1) / 2 * 400 + 50  # 50 à 450 N
            force_R = (math.sin(elapsed * 3 + math.pi) + 1) / 2 * 400 + 50
            
            # Simuler des pressions aléatoires sur les 16 capteurs
            # On divise la force totale grossièrement par le nombre de capteurs actifs
            pressures_L = [random.uniform(0, force_L/10) for _ in range(16)]
            pressures_R = [random.uniform(0, force_R/10) for _ in range(16)]

            # Créer le paquet de données simulé
            data = {
                "time": round(elapsed, 3),
                "left": {"pressure": pressures_L, "total_force": force_L},
                "right": {"pressure": pressures_R, "total_force": force_R},
            }
            
            try:
                self.data_queue.put_nowait(data)
            except queue.Full:
                pass
            
            time.sleep(0.05) # ~20 Hz

    # --- Réseau (UDP) ---
    @staticmethod
    def _parse_opengo_udp(raw_data: str):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_data)
        if len(nums) < 51: return None
        v = list(map(float, nums[:51]))
        return {
            "time_raw": v[0],
            "left": {"pressure": v[7:23], "total_force": v[25]},
            "right": {"pressure": v[32:48], "total_force": v[50]},
        }

    def _udp_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.UDP_IP, self.UDP_PORT))
        print(f"🔌 [LiveVisualizer] Écoute UDP sur {self.UDP_IP}:{self.UDP_PORT} ...")

        while True:
            data, _ = sock.recvfrom(self.BUFFER_SIZE)
            text = data.decode("utf-8", errors="ignore").strip()
            parsed = self._parse_opengo_udp(text)
            if not parsed: continue
                
            if self.t0 is None: self.t0 = parsed["time_raw"]
            parsed["time"] = round(parsed["time_raw"] - self.t0, 3)
            del parsed["time_raw"]
            
            try:
                self.data_queue.put_nowait(parsed)
            except queue.Full: pass

    def start_listener(self):
        if self.listener_thread is None:
            target_func = self._simulation_loop if self.simulation_mode else self._udp_listener
            self.listener_thread = threading.Thread(target=target_func, daemon=True)
            self.listener_thread.start()

    @staticmethod
    def _aggregate_zones(pressures):
        return {
            'Fore_Ext': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_FORE_EXT),
            'Fore_Int': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_FORE_INT),
            'Hind_Ext': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_HIND_EXT),
            'Hind_Int': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_HIND_INT),
        }

    def _calculate_radius(self, force):
        scaled_force = min(force, self.MAX_FORCE_N) / self.MAX_FORCE_N
        return self.BASE_RADIUS + (scaled_force * (self.MAX_RADIUS - self.BASE_RADIUS))

    def update_state(self):
        new_data = None
        try:
            while True: new_data = self.data_queue.get_nowait()
        except queue.Empty: pass

        if new_data is None and self.last_data_packet is None: return False
        if new_data is not None: self.last_data_packet = new_data
        
        data = self.last_data_packet
        self.current_time = data["time"]
        self.radius_L = self._calculate_radius(data["left"]["total_force"])
        self.radius_R = self._calculate_radius(data["right"]["total_force"])
        
        self.zones_L = self._aggregate_zones(data["left"]["pressure"])
        self.zones_R = self._aggregate_zones(data["right"]["pressure"])
        
        max_val = max(
            max(self.zones_L.values()) if self.zones_L else 0,
            max(self.zones_R.values()) if self.zones_R else 0,
            1.0
        )
        self.max_pressure = max_val
        self.norm = mcolors.Normalize(vmin=0, vmax=self.max_pressure)
        return True

    def get_radii(self):
        return (self.radius_L, self.radius_R)

    def get_zone_colors(self):
        colors_L = {k: self.cmap(self.norm(v)) for k, v in self.zones_L.items()}
        colors_R = {k: self.cmap(self.norm(v)) for k, v in self.zones_R.items()}
        return (colors_L, colors_R)


############################################################################
#
# --- SCRIPT PRINCIPAL (Visualisation 3D / Anneaux / Orthogonal) ---
#
############################################################################

# 1. ACTIVER LE MODE SIMULATION ICI
SIMULATION = True 

visualizer = LivePressureVisualizer(simulation_mode=SIMULATION)
visualizer.start_listener()

# Configuration MediaPipe
model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker_full.task')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modèle manquant: {model_path}")

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# Paramètres graphiques
BASE_RADIUS_PX = 30
MAX_RADIUS_PX = 100
RING_HOLE_RATIO = 0.6

def rgba_to_bgr(rgba):
    r, g, b, a = rgba
    return (int(b * 255), int(g * 255), int(r * 255))

def calculate_leg_angle(knee_pt, ankle_pt):
    """Angle du vecteur tibia par rapport à l'horizontale."""
    dx = ankle_pt[0] - knee_pt[0]
    dy = ankle_pt[1] - knee_pt[1]
    return math.degrees(math.atan2(dy, dx))

# --- 4. Fonctions de Dessin avec Transparence ---

def draw_foot_pressure_ring_3d(target_image, original_bg, center_pt, knee_pt, ankle_pt, normalized_radius, colors_rgba, is_right=False):
    """
    Dessine un ANNEAU avec centre transparent.
    :param target_image: L'image sur laquelle on dessine (sera modifiée).
    :param original_bg: Une copie propre de l'image de fond (pour restaurer le trou).
    """
    cx, cy = center_pt
    
    # 1. Calculs de géométrie (taille et perspective)
    norm_range = visualizer.MAX_RADIUS - visualizer.BASE_RADIUS
    pixel_range = MAX_RADIUS_PX - BASE_RADIUS_PX
    radius_ratio = (normalized_radius - visualizer.BASE_RADIUS) / norm_range
    radius_px_major = int(BASE_RADIUS_PX + (radius_ratio * pixel_range))
    
    if radius_px_major <= 0: return

    # Perspective orthogonale au tibia
    leg_angle_deg = calculate_leg_angle(knee_pt, ankle_pt)
    ellipse_rotation = leg_angle_deg + 90 
    radius_px_minor = int(radius_px_major * 0.4) # Ratio perspective 0.4
    axes_outer = (radius_px_major, radius_px_minor)
    
    # Conversion couleurs
    c_colors = {k: rgba_to_bgr(v) for k, v in colors_rgba.items()}

    # Définition des quadrants
    if not is_right: # Pied Gauche
        zones = [
            (c_colors['Fore_Int'], 0, 90),
            (c_colors['Fore_Ext'], 90, 180),
            (c_colors['Hind_Ext'], 180, 270),
            (c_colors['Hind_Int'], 270, 360),
        ]
    else: # Pied Droit
        zones = [
            (c_colors['Fore_Int'], 90, 180),   
            (c_colors['Fore_Ext'], 0, 90),     
            (c_colors['Hind_Ext'], 270, 360),  
            (c_colors['Hind_Int'], 180, 270),  
        ]

    # 2. Dessiner le "Disque Plein" Coloré
    # On dessine d'abord tout le disque sur l'image cible
    for color, start, end in zones:
        cv2.ellipse(target_image, (cx, cy), axes_outer, ellipse_rotation, start, end, color, -1)

    # 3. Créer le "Trou" Transparent
    # On calcule la taille du trou
    r_hole_maj = int(radius_px_major * RING_HOLE_RATIO)
    r_hole_min = int(radius_px_minor * RING_HOLE_RATIO)
    
    if r_hole_maj > 0:
        # A. Création d'un masque pour le trou (blanc = trou, noir = reste)
        mask = np.zeros(target_image.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (r_hole_maj, r_hole_min), ellipse_rotation, 0, 360, 255, -1)
        
        # B. Restauration du fond : Là où le masque est blanc, on remet les pixels de l'image originale
        # Cela efface la couleur qu'on vient de mettre au centre
        target_image[mask == 255] = original_bg[mask == 255]

        # C. (Optionnel) Ajouter une fine bordure pour délimiter le trou
        cv2.ellipse(target_image, (cx, cy), (r_hole_maj, r_hole_min), ellipse_rotation, 0, 360, (50,50,50), 1)
        cv2.ellipse(target_image, (cx, cy), axes_outer, ellipse_rotation, 0, 360, (50,50,50), 1)


# --- 5. Initialisation de la webcam ---
cap = cv2.VideoCapture(0) # Attention: remettre 0 si 1 ne fonctionne pas
cap.set(3, 640)
cap.set(4, 480)

print("Démarrage... En attente de données.")
while not visualizer.update_state():
    time.sleep(0.1)
print("Données reçues ! Affichage.")

# --- 6. Boucle Principale ---
try:
    while cap.isOpened():
        visualizer.update_state()
        r_L, r_R = visualizer.get_radii()
        c_L, c_R = visualizer.get_zone_colors()
        
        success, image = cap.read()
        if not success: continue

        # SAUVEGARDE DU FOND ORIGINAL (Important pour la transparence)
        original_background = image.copy()

        # Détection MediaPipe
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        res = detector.detect_for_video(mp_img, ts)

        # On dessine sur 'image' qui deviendra le résultat final
        annotated = image 

        for pose in res.pose_landmarks:
            h, w = image.shape[:2]
            try:
                # --- Pied Gauche ---
                lk, la = pose[25], pose[27]
                l_knee = (int(lk.x*w), int(lk.y*h))
                l_ankle = (int(la.x*w), int(la.y*h))
                lh, lt = pose[29], pose[31]
                l_cx = int((lh.x + lt.x)*w/2)
                l_cy = int((lh.y + lt.y)*h/2)

                draw_foot_pressure_ring_3d(
                    annotated, original_background, # On passe les deux images
                    (l_cx, l_cy + 15), l_knee, l_ankle, r_L, c_L, False
                )

                # --- Pied Droit ---
                rk, ra = pose[26], pose[28]
                r_knee = (int(rk.x*w), int(rk.y*h))
                r_ankle = (int(ra.x*w), int(ra.y*h))
                rh, rt = pose[30], pose[32]
                r_cx = int((rh.x + rt.x)*w/2)
                r_cy = int((rh.y + rt.y)*h/2)

                draw_foot_pressure_ring_3d(
                    annotated, original_background, # On passe les deux images
                    (r_cx, r_cy + 15), r_knee, r_ankle, r_R, c_R, True
                )
            except IndexError: continue

        cv2.imshow('Pression Anneaux Transparents', annotated)
        if cv2.waitKey(5) & 0xFF == 27: break

finally:
    cap.release()
    cv2.destroyAllWindows()
import socket
import re
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import time

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
    BASE_RADIUS = 0.2
    MAX_RADIUS = 1.0
    RING_WIDTH = 0.1
    CMAP_NAME = 'YlOrBr'

    def __init__(self):
        # --- File de communication ---
        self.data_queue = queue.Queue(maxsize=self.QUEUE_MAX)
        self.t0 = None
        self.listener_thread = None
        
        # --- État de la visualisation (accessible via les getters) ---
        self.last_data_packet = None
        self.current_time = 0.0
        self.radius_L = self.BASE_RADIUS
        self.radius_R = self.BASE_RADIUS
        self.center_L = (-1.2, 0)
        self.center_R = (1.2, 0)
        
        # Initialisation des zones à 0
        self.zones_L = self._get_empty_zones()
        self.zones_R = self._get_empty_zones()
        
        # --- Configuration Matplotlib (Couleurs) ---
        self.max_pressure = 1.0  # Normalisation dynamique
        self.norm = mcolors.Normalize(vmin=0, vmax=self.max_pressure)
        self.cmap = cm.get_cmap(self.CMAP_NAME)

    def _get_empty_zones(self):
        """Retourne un dictionnaire de zones initialisé à 0."""
        return {'Fore_Int': 0, 'Fore_Ext': 0, 'Hind_Ext': 0, 'Hind_Int': 0}

    # === Parsing (méthode statique car elle ne dépend pas de l'état) ===
    @staticmethod
    def _parse_opengo_udp(raw_data: str):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_data)
        if len(nums) < 51:
            return None
        v = list(map(float, nums[:51]))
        return {
            "time_raw": v[0],
            "left": {
                "pressure": v[7:23],
                "total_force": v[25],
            },
            "right": {
                "pressure": v[32:48],
                "total_force": v[50],
            },
        }

    # === Agrégation (méthode statique) ===
    @staticmethod
    def _aggregate_zones(pressures):
        return {
            'Fore_Ext': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_FORE_EXT),
            'Fore_Int': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_FORE_INT),
            'Hind_Ext': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_HIND_EXT),
            'Hind_Int': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_HIND_INT),
        }

    # === Thread d’écoute UDP ===
    def _udp_listener(self):
        """Cible du thread : écoute l'UDP et remplit la file d'attente."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.UDP_IP, self.UDP_PORT))
        print(f"🔌 Écoute UDP sur {self.UDP_IP}:{self.UDP_PORT} ...")

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
                pass # Ignorer les paquets si la file est pleine

    # === Méthode utilitaire pour le rayon ===
    def _calculate_radius(self, force):
        """Calcule le rayon en fonction de la force."""
        scaled_force = min(force, self.MAX_FORCE_N) / self.MAX_FORCE_N
        return self.BASE_RADIUS + (scaled_force * (self.MAX_RADIUS - self.BASE_RADIUS))

    # === Démarrage ===
    def start_listener(self):
        """Démarre le thread d'écoute UDP."""
        if self.listener_thread is None:
            self.listener_thread = threading.Thread(target=self._udp_listener, daemon=True)
            self.listener_thread.start()
            print("Thread d'écoute démarré.")

    # === Mise à jour de l'état ===
    def update_state(self):
        """
        Met à jour l'état interne en vidant la file d'attente.
        À appeler à chaque frame de votre boucle de visualisation.
        Retourne True si l'état a été mis à jour, False sinon.
        """
        new_data = None
        try:
            # Vider la file pour obtenir la donnée la plus récente
            while True:
                new_data = self.data_queue.get_nowait()
        except queue.Empty:
            pass # Pas de nouvelle donnée, on garde la dernière

        if new_data is None and self.last_data_packet is None:
            # Aucune donnée reçue
            return False
        
        if new_data is not None:
            self.last_data_packet = new_data
        
        # Mettre à jour toutes les variables d'état
        data = self.last_data_packet
        
        self.current_time = data["time"]
        self.radius_L = self._calculate_radius(data["left"]["total_force"])
        self.radius_R = self._calculate_radius(data["right"]["total_force"])
        
        self.zones_L = self._aggregate_zones(data["left"]["pressure"])
        self.zones_R = self._aggregate_zones(data["right"]["pressure"])
        
        # Mettre à jour la normalisation des couleurs (dynamique)
        max_L = max(self.zones_L.values())
        max_R = max(self.zones_R.values())
        self.max_pressure = max(max_L, max_R, 1.0) # Évite la division par 0
        self.norm = mcolors.Normalize(vmin=0, vmax=self.max_pressure)
        
        return True

    # --- GETTERS PUBLICS ---

    def get_current_time(self):
        """Retourne le temps (en s) de la dernière donnée traitée."""
        return self.current_time

    def get_radii(self):
        """Retourne un tuple (rayon_gauche, rayon_droit)."""
        return (self.radius_L, self.radius_R)

    def get_centers(self):
        """Retourne un tuple (centre_gauche, centre_droit)."""
        return (self.center_L, self.center_R)

    def get_ring_width(self):
        """Retourne l'épaisseur de l'anneau."""
        return self.RING_WIDTH

    def get_zone_pressures(self):
        """
        Retourne les dictionnaires de pression agrégée.
        (zones_gauche, zones_droit)
        """
        return (self.zones_L, self.zones_R)

    def get_zone_colors(self):
        """
        Calcule et retourne les couleurs (RGBA) pour chaque quadrant.
        Retourne (couleurs_gauche, couleurs_droit)
        """
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

    def get_last_data_packet(self):
        """Retourne le dictionnaire de données brutes parsées complet."""
        return self.last_data_packet
        
    def get_colormap_config(self):
        """Retourne la colormap et la normalisation pour une colorbar."""
        return self.cmap, self.norm, self.max_pressure


# === EXEMPLE D'UTILISATION (Identique à votre script original) ===

if __name__ == "__main__":
    
    FPS = 20 # Taux de rafraîchissement de l'animation

    # 1. Initialiser la classe
    visualizer = LivePressureVisualizer()
    
    # 2. Démarrer l'écoute UDP
    visualizer.start_listener()

    # 3. Configurer la figure Matplotlib
    fig, ax = plt.subplots(figsize=(12, 7))

    # Attendre les premières données pour éviter un affichage vide
    print("En attente des premières données UDP...")
    while not visualizer.update_state():
        time.sleep(0.1)
    print("Données reçues. Démarrage de la visualisation.")

    def animate(_):
        """
        Fonction d'animation qui utilise les getters du visualizer.
        """
        
        # 1. Mettre à jour l'état interne du visualizer
        updated = visualizer.update_state()
        
        # if not updated:
        #     return # Optionnel: ne pas redessiner si pas de nouvelle donnée

        # 2. Récupérer les données via les getters
        t = visualizer.get_current_time()
        radius_L, radius_R = visualizer.get_radii()
        center_L, center_R = visualizer.get_centers()
        colors_L, colors_R = visualizer.get_zone_colors()
        width = visualizer.get_ring_width()

        # 3. Dessiner (identique à votre script)
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')

        # Pied gauche
        ax.add_patch(patches.Wedge(center_L, radius_L, 0, 90, width=width, facecolor=colors_L['Fore_Int']))
        ax.add_patch(patches.Wedge(center_L, radius_L, 90, 180, width=width, facecolor=colors_L['Fore_Ext']))
        ax.add_patch(patches.Wedge(center_L, radius_L, 180, 270, width=width, facecolor=colors_L['Hind_Ext']))
        ax.add_patch(patches.Wedge(center_L, radius_L, 270, 360, width=width, facecolor=colors_L['Hind_Int']))

        # Pied droit
        ax.add_patch(patches.Wedge(center_R, radius_R, 0, 90, width=width, facecolor=colors_R['Fore_Int']))
        ax.add_patch(patches.Wedge(center_R, radius_R, 90, 180, width=width, facecolor=colors_R['Fore_Ext']))
        ax.add_patch(patches.Wedge(center_R, radius_R, 180, 270, width=width, facecolor=colors_R['Hind_Ext']))
        ax.add_patch(patches.Wedge(center_R, radius_R, 270, 360, width=width, facecolor=colors_R['Hind_Int']))

        # Titres et labels
        ax.text(-1.2, 1.3, 'PIED GAUCHE', ha='center', fontsize=12, fontweight='bold')
        ax.text(1.2, 1.3, 'PIED DROIT', ha='center', fontsize=12, fontweight='bold')
        fig.suptitle(f'Visualisation de la Pression - Temps : {t:.2f} s', fontsize=16, y=0.95)

        # Labels (simplifiés pour la lisibilité)
        ax.text(-1.2, 0.8, 'Avant', ha='center', fontsize=8, alpha=0.5)
        ax.text(-1.2, -0.8, 'Arrière', ha='center', fontsize=8, alpha=0.5)
        ax.text(1.2, 0.8, 'Avant', ha='center', fontsize=8, alpha=0.5)
        ax.text(1.2, -0.8, 'Arrière', ha='center', fontsize=8, alpha=0.5)

    # 4. Lancer l'animation
    print("🎥 Visualisation live prête.")
    ani = animation.FuncAnimation(fig, animate, interval=int(1000/FPS), blit=False)
    plt.show()
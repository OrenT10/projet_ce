import socket
import re
import threading
import queue
import time
import math
import random
import matplotlib.colors as mcolors
import matplotlib.cm as cm

class LivePressureVisualizer:
    """
    Gère la réception (UDP) ou la simulation des données de pression
    et convertit ces données en couleurs et rayons pour la visualisation.
    """

    # --- Configuration UDP ---
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    BUFFER_SIZE = 2048
    QUEUE_MAX = 1000

    # --- Mapping Capteurs ---
    SENSORS_FORE_EXT = [8, 11, 12, 13, 16]
    SENSORS_FORE_INT = [7, 9, 10, 14, 15]
    SENSORS_HIND_EXT = [2, 4, 6]
    SENSORS_HIND_INT = [1, 3, 5]

    # --- Configuration Données ---
    MAX_FORCE_N = 500.0
    BASE_RADIUS = 0.2
    MAX_RADIUS = 1.0
    CMAP_NAME = 'YlOrBr'

    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.data_queue = queue.Queue(maxsize=self.QUEUE_MAX)
        self.t0 = None
        self.listener_thread = None
        
        self.last_data_packet = None
        self.radius_L = self.BASE_RADIUS
        self.radius_R = self.BASE_RADIUS
        
        self.zones_L = self._get_empty_zones()
        self.zones_R = self._get_empty_zones()
        
        self.max_pressure = 1.0
        self.norm = mcolors.Normalize(vmin=0, vmax=self.max_pressure)
        self.cmap = cm.get_cmap(self.CMAP_NAME)

    def _get_empty_zones(self):
        return {'Fore_Int': 0, 'Fore_Ext': 0, 'Hind_Ext': 0, 'Hind_Int': 0}

    def start_listener(self):
        """Démarre le thread d'écoute (UDP ou Simulation)."""
        if self.listener_thread is None:
            target_func = self._simulation_loop if self.simulation_mode else self._udp_listener
            self.listener_thread = threading.Thread(target=target_func, daemon=True)
            self.listener_thread.start()
            mode = "SIMULATION" if self.simulation_mode else f"UDP ({self.UDP_PORT})"
            print(f"🔌 [Visualizer] Thread démarré en mode : {mode}")

    def update_state(self):
        """Met à jour l'état interne avec les dernières données reçues."""
        new_data = None
        try:
            while True: new_data = self.data_queue.get_nowait()
        except queue.Empty: pass

        if new_data is None and self.last_data_packet is None: return False
        if new_data is not None: self.last_data_packet = new_data
        
        data = self.last_data_packet
        
        # Mise à jour rayons et couleurs
        self.radius_L = self._calculate_radius(data["left"]["total_force"])
        self.radius_R = self._calculate_radius(data["right"]["total_force"])
        self.zones_L = self._aggregate_zones(data["left"]["pressure"])
        self.zones_R = self._aggregate_zones(data["right"]["pressure"])
        
        # Normalisation dynamique des couleurs
        max_val = max(
            max(self.zones_L.values() or [0]),
            max(self.zones_R.values() or [0]),
            1.0
        )
        self.norm = mcolors.Normalize(vmin=0, vmax=max_val)
        return True

    def get_radii(self):
        return (self.radius_L, self.radius_R)

    def get_zone_colors(self):
        """Retourne les couleurs RGBA pour l'affichage."""
        colors_L = {k: self.cmap(self.norm(v)) for k, v in self.zones_L.items()}
        colors_R = {k: self.cmap(self.norm(v)) for k, v in self.zones_R.items()}
        return (colors_L, colors_R)

    # --- Méthodes Internes (Simulation & Parsing) ---
    def _simulation_loop(self):
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            # Sinusoïde en opposition de phase
            force_L = (math.sin(elapsed * 3) + 1) / 2 * 400 
            force_R = (math.sin(elapsed * 3 + math.pi) + 1) / 2 * 400 
            pressures_L = [0, 0, 0, 0, 0, 0, force_L/10, force_L/10, force_L/10, force_L/10, force_L/10, force_L/10, force_L/10, force_L/10, force_L/10, force_L/10]
            pressures_R = [0, 0, 0, 0, 0, 0, force_R/10, force_R/10, force_R/10, force_R/10, force_R/10, force_R/10, force_R/10, force_R/10, force_R/10, force_R/10]
            
            self.data_queue.put({
                "time": round(elapsed, 3),
                "left": {"pressure": pressures_L, "total_force": force_L},
                "right": {"pressure": pressures_R, "total_force": force_R},
            })
            time.sleep(0.05)

    def _udp_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.UDP_IP, self.UDP_PORT))
        while True:
            data, _ = sock.recvfrom(self.BUFFER_SIZE)
            text = data.decode("utf-8", errors="ignore").strip()
            parsed = self._parse_opengo_udp(text)
            if parsed:
                if self.t0 is None: self.t0 = parsed["time_raw"]
                parsed["time"] = round(parsed["time_raw"] - self.t0, 3)
                del parsed["time_raw"]
                try: self.data_queue.put_nowait(parsed)
                except queue.Full: pass

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

    @staticmethod
    def _aggregate_zones(pressures):
        return {
            'Fore_Ext': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_FORE_EXT),
            'Fore_Int': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_FORE_INT),
            'Hind_Ext': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_HIND_EXT),
            'Hind_Int': sum(pressures[i-1] for i in LivePressureVisualizer.SENSORS_HIND_INT),
        }

    def _calculate_radius(self, force):
        scaled = min(force, self.MAX_FORCE_N) / self.MAX_FORCE_N
        return self.BASE_RADIUS + (scaled * (self.MAX_RADIUS - self.BASE_RADIUS))
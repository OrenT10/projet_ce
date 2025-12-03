import socket
import threading
import queue
import time
import math
import random
from dataclasses import dataclass, field
from typing import List

@dataclass
class FootData:
    acceleration: List[float] = field(default_factory=lambda: [0.0]*3)
    angular: List[float] = field(default_factory=lambda: [0.0]*3)
    cop: List[float] = field(default_factory=lambda: [0.0]*2)
    pressure: List[float] = field(default_factory=lambda: [0.0]*16)
    total_force: float = 0.0

@dataclass
class SensorData:
    timestamp: float = 0.0
    right_total_force_idx1: float = 0.0 # Le "0.righttotal_force" étrange du début
    left: FootData = field(default_factory=FootData)
    right: FootData = field(default_factory=FootData)

class DataManager:
    """
    Responsable de l'acquisition des données (UDP ou Simu) 
    et de leur normalisation au format SensorData.
    """
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    BUFFER_SIZE = 4096

    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.data_queue = queue.Queue(maxsize=1) # On garde seulement la dernière frame
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        target = self._simulation_loop if self.simulation_mode else self._udp_listener
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()
        print(f"[DataManager] Démarré en mode {'SIMULATION' if self.simulation_mode else 'UDP'}")

    def get_latest_data(self):
        """Récupère la dernière trame disponible sans bloquer."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    # --- SIMULATION ---
    def _simulation_loop(self):
        t0 = time.time()
        while self.running:
            t = time.time() - t0
            # Simulation d'un balancement gauche/droite
            force_L = (math.sin(t * 2) + 1) * 150 + 20
            force_R = (math.sin(t * 2 + math.pi) + 1) * 150 + 20
            
            # Simulation de pressions (répartition simple)
            pressures_L = [force_L / 16.0] * 16
            pressures_R = [force_R / 16.0] * 16
            
            # Simulation d'un COP qui bouge un peu (instabilité simulée)
            cop_x = math.sin(t * 5) * 0.5
            cop_y = math.cos(t * 3) * 0.5

            data = SensorData(
                timestamp=t,
                right_total_force_idx1=force_R,
                left=FootData(cop=[cop_x, cop_y], pressure=pressures_L, total_force=force_L),
                right=FootData(cop=[-cop_x, -cop_y], pressure=pressures_R, total_force=force_R)
            )
            
            self._push_data(data)
            time.sleep(0.02) # ~50Hz

    # --- UDP ---
    def _udp_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.UDP_IP, self.UDP_PORT))
        sock.settimeout(1.0)
        
        print(f"[UDP] Écoute sur {self.UDP_PORT}...")
        
        while self.running:
            try:
                raw_data, _ = sock.recvfrom(self.BUFFER_SIZE)
                decoded = raw_data.decode('utf-8').strip()
                parsed_data = self._parse_string_52(decoded)
                if parsed_data:
                    self._push_data(parsed_data)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[UDP Error] {e}")

    def _push_data(self, data):
        # On remplace la vieille donnée si la queue est pleine pour toujours avoir la + récente
        if self.data_queue.full():
            try: self.data_queue.get_nowait()
            except: pass
        self.data_queue.put(data)

    def _parse_string_52(self, text_line):
        """Parse la ligne de 52 valeurs spécifiée."""
        # Séparer par espace
        parts = text_line.split()
        if len(parts) < 52:
            return None # Trame incomplète
        
        try:
            vals = [float(x) for x in parts]
            
            # Mapping exact selon votre ordre :
            # 0: timestamp
            # 1: 0.righttotal_force
            # 2-4: left.acc
            # 5-7: left.ang
            # 8-9: left.cop
            # 10-25: left.pressure (16)
            # 26: left.total_force
            # ... et idem pour right ...
            
            d = SensorData()
            d.timestamp = vals[0]
            d.right_total_force_idx1 = vals[1]
            
            # Left
            d.left.acceleration = vals[2:5]
            d.left.angular = vals[5:8]
            d.left.cop = vals[8:10]
            d.left.pressure = vals[10:26]
            d.left.total_force = vals[26]
            
            # Right (Indices décalés de 27)
            offset = 27
            d.right.acceleration = vals[offset:offset+3]
            d.right.angular = vals[offset+3:offset+6]
            d.right.cop = vals[offset+6:offset+8]
            d.right.pressure = vals[offset+8:offset+24]
            d.right.total_force = vals[offset+24]
            
            return d
        except ValueError:
            return None
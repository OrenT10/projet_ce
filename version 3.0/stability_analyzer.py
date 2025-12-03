import numpy as np
import collections

class StabilityAnalyzer:
    """
    Calcule un indice de stabilité.
    Ignore les données si le timestamp ne change pas (doublons).
    """
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.timestamps = collections.deque(maxlen=buffer_size)
        self.global_cops = collections.deque(maxlen=buffer_size) 
        
        self.stability_score = 100.0 
        self.status = "INIT"
        self.last_timestamp = -1.0 # <-- Pour éviter les doublons

    def update(self, sensor_data):
        if not sensor_data: return
        
        # 1. Protection contre les doublons (dt = 0)
        if sensor_data.timestamp == self.last_timestamp:
            return # On ne recalcule pas si la donnée n'a pas changé
        
        self.last_timestamp = sensor_data.timestamp

        # 2. Calcul du COP Global
        F_L = sensor_data.left.total_force
        F_R = sensor_data.right.total_force
        Total_F = F_L + F_R
        
        # Seuil minimal pour considérer qu'il y a quelqu'un
        if Total_F < 5.0: 
            self.stability_score = 0
            self.status = "NO_USER"
            # On reset les buffers pour éviter de lier des mouvements distants dans le temps
            self.timestamps.clear()
            self.global_cops.clear()
            return

        g_cop_x = (sensor_data.left.cop[0] * F_L + sensor_data.right.cop[0] * F_R) / Total_F
        g_cop_y = (sensor_data.left.cop[1] * F_L + sensor_data.right.cop[1] * F_R) / Total_F

        self.timestamps.append(sensor_data.timestamp)
        self.global_cops.append((g_cop_x, g_cop_y))

        # 3. Analyse
        if len(self.global_cops) >= 5: # Au moins quelques points
            self._compute_metrics()

    def _compute_metrics(self):
        cops = np.array(self.global_cops)
        times = np.array(self.timestamps)
        
        # Vitesse
        dists = np.sqrt(np.diff(cops[:,0])**2 + np.diff(cops[:,1])**2)
        dts = np.diff(times)
        
        # Protection division par zero (normalement géré par le check timestamp, mais sécurité)
        dts[dts <= 0.0001] = 0.001 
        
        velocities = dists / dts
        avg_velocity = np.mean(velocities)

        # Aire (Variance)
        std_x = np.std(cops[:,0])
        std_y = np.std(cops[:,1])
        area_approx = std_x * std_y * 1000 # *1000 pour mettre à l'échelle lisible

        # Score heuristique
        # Ajustez ces seuils selon vos tests réels
        penalty_vel = min(avg_velocity * 5.0, 50.0) 
        penalty_area = min(area_approx * 5.0, 50.0)
        
        raw_score = 100 - (penalty_vel + penalty_area)
        self.stability_score = max(0.0, min(100.0, raw_score))
        
        if self.stability_score > 75: self.status = "STABLE"
        elif self.stability_score > 40: self.status = "INSTABLE"
        else: self.status = "CRITIQUE"

    def get_stability_info(self):
        return self.stability_score, self.status
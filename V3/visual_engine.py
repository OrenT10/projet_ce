import cv2
import math
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

class PressureRing:
    """Gère le dessin géométrique de l'anneau 3D transparent."""
    BASE_RADIUS_PX = 40
    MAX_RADIUS_PX = 150 # <-- RÉDUIT pour éviter les ellipses géantes
    
    def _rgba_to_bgr(self, rgba):
        return (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

    def draw_on_overlay(self, overlay, center, heel, toe, radius_norm, colors, is_right):
        """Dessine l'anneau sur un calque noir (overlay) pour gérer la transparence."""
        if radius_norm <= 0: return

        # 1. Géométrie
        dx, dy = toe[0] - heel[0], toe[1] - heel[1]
        foot_angle = math.degrees(math.atan2(dy, dx))
        
        # Mapping de 0..1 vers BASE..MAX
        r_px = int(self.BASE_RADIUS_PX + (radius_norm * (self.MAX_RADIUS_PX - self.BASE_RADIUS_PX)))
        axes = (r_px, int(r_px * 0.35)) # Perspective aplatie

        # 2. Zones (Mapping des couleurs)
        zone_defs = []
        if is_right:
            zone_defs = [
                (colors['Fore_Int'], 0, 90), 
                (colors['Fore_Ext'], 270, 360),    
                (colors['Hind_Ext'], 180, 270),  
                (colors['Hind_Int'], 90, 180)
            ]
        else:
            zone_defs = [
                (colors['Fore_Int'], 270, 360),    
                (colors['Fore_Ext'], 0, 90), 
                (colors['Hind_Ext'], 90, 180), 
                (colors['Hind_Int'], 180, 270)
            ]

        # 3. Dessin des secteurs PLEINS sur le calque
        for col_rgba, s, e in zone_defs:
            color_bgr = self._rgba_to_bgr(col_rgba)
            rot_s = (s + foot_angle) % 360
            rot_e = (e + foot_angle) % 360
            
            if rot_e < rot_s:
                cv2.ellipse(overlay, center, axes, 0, rot_s, 360, color_bgr, -1)
                cv2.ellipse(overlay, center, axes, 0, 0, rot_e, color_bgr, -1)
            else:
                cv2.ellipse(overlay, center, axes, 0, rot_s, rot_e, color_bgr, -1)

        # 4. Le "Trou" Central
        # On dessine un cercle NOIR (0,0,0) au centre du calque.
        # Lors de l'addition (Overlay + Frame), Noir + Image = Image (Transparence)
        hole_axes = (int(axes[0]*0.6), int(axes[1]*0.6))
        cv2.ellipse(overlay, center, hole_axes, 0, 0, 360, (0,0,0), -1) 
        
        # Contours esthétiques (blanc cassé)
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (200,200,200), 1)
        cv2.ellipse(overlay, center, hole_axes, 0, 0, 360, (200,200,200), 1)

class Visualizer:
    """Orchestre le rendu visuel."""
    
    # Indices capteurs (1-16 -> 0-15) inchangés
    S_FORE_EXT = [7, 10, 11, 12, 15] 
    S_FORE_INT = [6, 8, 9, 13, 14]   
    S_HIND_EXT = [1, 3, 5]           
    S_HIND_INT = [0, 2, 4]           

    def __init__(self, max_force=350.0):
        self.max_force = max_force
        self.ring_drawer = PressureRing()
        self.cmap = cm.get_cmap('YlOrBr')
        self.norm = mcolors.Normalize(vmin=0, vmax=1.0)

    def process_and_draw(self, frame, sensor_data, landmarks_result):
        if not sensor_data: return

        # 1. Calcul des données de couleur/rayon
        viz_data = {
            'left': self._compute_foot_viz(sensor_data.left),
            'right': self._compute_foot_viz(sensor_data.right)
        }

        # Update dynamique de la norme couleur
        all_vals = list(viz_data['left']['zones'].values()) + list(viz_data['right']['zones'].values())
        if all_vals:
            # On fixe un min à 0.1 pour éviter division par zero si tout est vide
            self.norm = mcolors.Normalize(vmin=0, vmax=max(max(all_vals), 0.1))

        # 2. Dessin avec Transparence
        if landmarks_result and landmarks_result.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks = landmarks_result.pose_landmarks[0]
            
            # CRÉATION DU CALQUE (OVERLAY)
            overlay = np.zeros_like(frame)
            
            self._draw_foot_on_overlay(overlay, landmarks, False, w, h, viz_data['left'])
            self._draw_foot_on_overlay(overlay, landmarks, True, w, h, viz_data['right'])
            
            # FUSION : Frame + Overlay * 0.6 (Transparence)
            cv2.addWeighted(overlay, 0.7, frame, 1.0, 0, frame)

    def _compute_foot_viz(self, foot_data):
        ratio = min(foot_data.total_force, self.max_force) / self.max_force
        p = foot_data.pressure
        zones = {
            'Fore_Ext': sum(p[i] for i in self.S_FORE_EXT),
            'Fore_Int': sum(p[i] for i in self.S_FORE_INT),
            'Hind_Ext': sum(p[i] for i in self.S_HIND_EXT),
            'Hind_Int': sum(p[i] for i in self.S_HIND_INT),
        }
        return {'ratio': ratio, 'zones': zones}

    def _draw_foot_on_overlay(self, overlay, landmarks, is_right, w, h, viz_data):
        idx_offset = 1 if is_right else 0
        try:
            ankle = landmarks[27 + idx_offset]
            heel = landmarks[29 + idx_offset]
            toe = landmarks[31 + idx_offset]
            
            p_heel = (int(heel.x * w), int(heel.y * h))
            p_toe = (int(toe.x * w), int(toe.y * h))
            p_ankle = (int(ankle.x * w), int(ankle.y * h))
            
            center_x = p_ankle[0]
            center_y = max(p_heel[1], p_toe[1]) + 15
            center = (center_x, center_y)

            colors = {k: self.cmap(self.norm(v)) for k,v in viz_data['zones'].items()}

            self.ring_drawer.draw_on_overlay(overlay, center, p_heel, p_toe, viz_data['ratio'], colors, is_right)
        except IndexError:
            pass
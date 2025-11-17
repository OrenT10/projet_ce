import cv2
import math
import numpy as np
from pressure_manager import LivePressureVisualizer

class PressureRing:
    """Gère le dessin géométrique de l'anneau 3D transparent."""
    
    BASE_RADIUS_PX = 5
    MAX_RADIUS_PX = 100
    RING_HOLE_RATIO = 0.7
    PERSPECTIVE_RATIO = 0.3

    @staticmethod
    def _rgba_to_bgr(rgba):
        r, g, b, a = rgba
        return (int(b * 255), int(g * 255), int(r * 255))

    @staticmethod
    def _calculate_angle(heel, toe):
        dx, dy = toe[0] - heel[0], toe[1] - heel[1]
        return math.degrees(math.atan2(dy, dx))

    def draw(self, target_image, original_bg, center, heel, toe, radius_norm, colors, is_right):
        # 1. Calcul de l'angle du pied pour faire tourner les COULEURS
        dx = toe[0] - heel[0]
        dy = toe[1] - heel[1]
        foot_angle = math.degrees(math.atan2(dy, dx))

        # 2. Géométrie "Assiette au sol" (Fixe)
        norm_range = LivePressureVisualizer.MAX_RADIUS - LivePressureVisualizer.BASE_RADIUS
        px_range = self.MAX_RADIUS_PX - self.BASE_RADIUS_PX
        ratio = (radius_norm - LivePressureVisualizer.BASE_RADIUS) / norm_range
        r_base = int(self.BASE_RADIUS_PX + (ratio * px_range))
        if r_base <= 0: return

        # La forme de l'ellipse est TOUJOURS horizontale (angle 0)
        # r_major est la largeur (X), r_minor est la profondeur (Y) écrasée par la perspective
        r_major = r_base
        r_minor = int(r_base * 0.35) # Ratio fixe pour simuler le sol
        axes = (r_major, r_minor)

        # 3. Couleurs
        c = {k: self._rgba_to_bgr(v) for k, v in colors.items()}

        # 4. Définition des Zones (Base + Rotation du pied)
        # On définit les angles de base comme si le pied regardait à droite (0°)
        # Puis on ajoute 'foot_angle' pour les faire tourner.
        
        base_zones = []
        if is_right:
            # (Couleur, Début Standard, Fin Standard)
            # Pour un pied droit regardant à 0° (Droite) :
            # Avant-Int (Haut-Gauche par rapport au pied) -> 270-360
            base_zones = [
                (c['Fore_Int'], 0, 90), 
                (c['Fore_Ext'], 270, 360),    
                (c['Hind_Ext'], 180, 270),  
                (c['Hind_Int'], 90, 180)  
            ]
        else:
            # Pied gauche
            base_zones = [
                (c['Fore_Int'], 270, 360),    
                (c['Fore_Ext'], 0, 90), 
                (c['Hind_Ext'], 90, 180), 
                (c['Hind_Int'], 180, 270)   
            ]

        # 5. Dessin : On applique la rotation aux angles, PAS à l'ellipse
        for col, s, e in base_zones:
            # On ajoute l'angle du pied aux secteurs
            # L'ellipse, elle, reste à angle=0
            
            rot_s = (s + foot_angle) % 360
            rot_e = (e + foot_angle) % 360
            
            # Gestion du passage par zéro (ex: 350° à 20°)
            if rot_e < rot_s:
                # Cas où on traverse 360 : on dessine en deux fois ou on laisse OpenCV gérer
                # OpenCV gère mal start > end, on fait donc :
                cv2.ellipse(target_image, center, axes, 0, rot_s, 360, col, -1)
                cv2.ellipse(target_image, center, axes, 0, 0, rot_e, col, -1)
            else:
                cv2.ellipse(target_image, center, axes, 0, rot_s, rot_e, col, -1)

        # 6. Transparence (Masque) - Forme fixe aussi
        r_hole_maj = int(r_major * self.RING_HOLE_RATIO)
        r_hole_min = int(r_minor * self.RING_HOLE_RATIO)
        hole_axes = (r_hole_maj, r_hole_min)
        
        if r_hole_maj > 0:
            mask = np.zeros(target_image.shape[:2], dtype=np.uint8)
            # Angle = 0 ici aussi
            cv2.ellipse(mask, center, hole_axes, 0, 0, 360, 255, -1)
            
            target_image[mask == 255] = original_bg[mask == 255]
            
            # Bordures
            cv2.ellipse(target_image, center, hole_axes, 0, 0, 360, (50,50,50), 1)
            cv2.ellipse(target_image, center, axes, 0, 0, 360, (50,50,50), 1)

class FootTracker:
    """Gère l'état d'un pied (position MediaPipe + données pression UDP)."""
    
    def __init__(self, is_right, drawer):
        self.is_right = is_right
        self.drawer = drawer
        # Indices MediaPipe
        self.knee_idx = 26 if is_right else 25
        self.ankle_idx = 28 if is_right else 27
        self.heel_idx = 30 if is_right else 29
        self.toe_idx = 32 if is_right else 31
        
        self.visible = False
        self.coords = {} 
        
        # (Radius, Colors). Données venant de l'UDP via update_pressure
        self.pressure_data = (0.2, {}) 
        
        # Mémoire de la position Y du sol (pour la perspective)
        self.floor_y = None 

    def update_landmarks(self, landmarks, w, h):
        try:
            lk, la = landmarks[self.knee_idx], landmarks[self.ankle_idx]
            lh, lt = landmarks[self.heel_idx], landmarks[self.toe_idx]
            
            # Conversion pixels
            knee = (int(lk.x * w), int(lk.y * h))
            ankle = (int(la.x * w), int(la.y * h))
            heel = (int(lh.x * w), int(lh.y * h))
            toe = (int(lt.x * w), int(lt.y * h))

            self.coords['knee'] = knee
            self.coords['ankle'] = ankle
            self.coords['heel'] = heel
            self.coords['toe'] = toe

            # --- LOGIQUE DE FUSION (VISION + UDP) ---
            
            # 1. Où est le bas du pied visuellement ?
            current_foot_bottom = max(heel[1], toe[1])
            
            # 2. Est-ce que les capteurs UDP détectent une pression ?
            # self.pressure_data[1] contient le dictionnaire des couleurs/zones actives
            has_pressure = bool(self.pressure_data[1])

            # Initialisation de la mémoire sol si c'est la première frame
            if self.floor_y is None:
                self.floor_y = current_foot_bottom
            
            # 3. Mise à jour de la mémoire du sol
            # Si les capteurs disent "contact" (has_pressure), alors la position actuelle
            # EST la position du sol (cela gère automatiquement la perspective).
            if has_pressure:
                self.floor_y = current_foot_bottom
            
            # Si has_pressure est False (pied levé), on NE touche PAS à self.floor_y.
            # L'anneau restera donc verrouillé à la dernière position connue du sol.

            # --- CALCUL DU CENTRE DE L'ANNEAU ---
            # X : Suit la cheville (l'axe du pied)
            center_x = ankle[0]
            
            # Y : Utilise la mémoire du sol (+ petit offset esthétique)
            # Que le pied soit levé ou posé, l'anneau reste au niveau du sol virtuel
            center_y = self.floor_y + 10
            
            self.coords['center'] = (center_x, center_y)
            self.visible = True
            
        except Exception as e:
            self.visible = False

    def update_pressure(self, radius, colors):
        """Reçoit les données UDP traitées depuis le script principal."""
        self.pressure_data = (radius, colors)
        
    def render(self, target, bg):
        # On dessine SEULEMENT si on a des données UDP (Pression > 0)
        # Si le pied est levé (colors vide), l'anneau disparaît naturellement.
        if self.visible and self.pressure_data[1]:
            self.drawer.draw(
                target, bg, 
                self.coords['center'], # Position ancrée au sol
                self.coords['heel'],
                self.coords['toe'], 
                self.pressure_data[0], self.pressure_data[1], 
                self.is_right
            )
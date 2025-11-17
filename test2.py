import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

# --- Configuration ---

# Nom du fichier de données
DATA_FILE = "raw_data/datas_for_processing.txt"

# Mappage des capteurs (basé sur votre description, index 1-based)
SENSORS_FORE_EXT = [8, 11, 12, 13, 16]
SENSORS_FORE_INT = [7, 9, 10, 14, 15]
SENSORS_HIND_EXT = [2, 4, 6]
SENSORS_HIND_INT = [1, 3, 5]

# Constantes pour l'animation
MAX_FORCE_N = 500.0  # 500N pour le rayon max
BASE_RADIUS = 0.2    # Rayon minimum (quand la force est 0)
MAX_RADIUS = 1.0     # Rayon maximum (quand la force est >= 500N)
RING_WIDTH = 0.1     # Épaisseur de l'anneau (proportionnelle au rayon)
CMAP_NAME = 'YlOrBr' # Palette de couleur (Yellow-Orange-Brown)

# --- Point 1 & 2 : Chargement et Traitement des Données ---

def load_and_process_data(file_path):
    """
    Charge le fichier .txt, le nettoie, et agrège les capteurs en 4 zones
    pour chaque pied.
    """
    print(f"Chargement du fichier de données : {file_path}...")
    try:
        # Charger les données en ignorant les 9 premières lignes (vrais commentaires)
        # et en utilisant '#' pour ignorer les commentaires en fin de ligne de la 10ème ligne.
        df = pd.read_csv(
            file_path, 
            sep='\t', 
            skiprows=9,  # Ignorer les 9 premières lignes (vrais commentaires)
        )
        
        # --- NOUVELLE ÉTAPE : Nettoyer les noms de colonnes ---
        # La 10ème ligne (maintenant la première) commence par '# ' mais décrit les colonnes 
        # On supprime le '#' et l'espace au début du nom de chaque colonne.
        df.columns = df.columns.str.lstrip('# ') 
        
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{file_path}' n'a pas été trouvé.")
        print("Assurez-vous qu'il se trouve dans le même dossier que le script.")
        return None
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture du fichier : {e}")
        return None

    # ... (le reste de votre fonction, qui ne change pas)

    print("Données chargées. Traitement et agrégation des capteurs...")

    # Dictionnaire pour mapper les noms de zones aux listes de capteurs
    zone_mappings = {
        'Fore_Ext': SENSORS_FORE_EXT,
        'Fore_Int': SENSORS_FORE_INT,
        'Hind_Ext': SENSORS_HIND_EXT,
        'Hind_Int': SENSORS_HIND_INT,
    }

    # Créer le nouveau DataFrame "processed"
    # On garde 'time' et les colonnes de force totale
    try:
        force_cols = ['left total force[N]', 'right total force[N]']
        df_processed = df[['time'] + force_cols].copy()

        # Agréger les capteurs pour chaque zone et chaque pied
        for side in ['left', 'right']:
            for zone_name, sensor_indices in zone_mappings.items():
                # Construire les noms de colonnes exacts
                col_names = [f'{side} pressure {i}[N/cm²]' for i in sensor_indices]
                
                # Créer le nom de la nouvelle colonne agrégée
                new_col_name = f'{side[0].upper()}_{zone_name}' # Ex: L_Fore_Ext
                
                # Calculer la somme des capteurs de cette zone
                df_processed[new_col_name] = df[col_names].sum(axis=1)

    except KeyError as e:
        print(f"ERREUR : Colonne non trouvée dans le fichier : {e}")
        print("Vérifiez que les en-têtes du fichier .txt correspondent.")
        return None

    print("Traitement terminé.")
    return df_processed

# --- Point 3 : Création de l'Animation ---

def create_animation(df):
    """
    Crée et affiche l'animation des cercles de pression.
    """
    print("Préparation de l'animation...")

    # Trouver la pression maximale pour normaliser l'échelle de couleur
    pressure_cols = [col for col in df.columns if col.startswith(('L_', 'R_'))]
    max_pressure = df[pressure_cols].max().max()
    if max_pressure == 0:
        print("Avertissement : Aucune donnée de pression trouvée. L'animation sera jaune.")
        max_pressure = 1.0 # Éviter la division par zéro

    # Configuration de la colormap (Jaune -> Marron)
    norm = mcolors.Normalize(vmin=0, vmax=max_pressure)
    cmap = cm.get_cmap(CMAP_NAME)

    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 7))

    def get_radius(force):
        """Fonction linéaire pour le rayon en fonction de la force."""
        # Limiter la force à 300N
        scaled_force = min(force, MAX_FORCE_N) / MAX_FORCE_N
        # Calculer le rayon (min 0.2, max 1.0)
        return BASE_RADIUS + (scaled_force * (MAX_RADIUS - BASE_RADIUS))

    def animate(i):
        """Fonction appelée pour chaque image de l'animation."""
        data = df.iloc[i]
        ax.clear() # Effacer l'image précédente
        
        # --- Configuration des axes ---
        ax.set_aspect('equal')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off') # Cacher les axes
        
        # --- Calcul des rayons ---
        radius_L = get_radius(data['left total force[N]'])
        radius_R = get_radius(data['right total force[N]'])
        
        # --- Définition des centres ---
        center_L = (-1.2, 0)
        center_R = (1.2, 0)

        # --- DESSIN DU PIED GAUCHE ---
        # Quadrant Avant-Interne (Top-Right)
        wedge = patches.Wedge(center_L, radius_L, 0, 90, width=RING_WIDTH, 
                              facecolor=cmap(norm(data['L_Fore_Int'])))
        ax.add_patch(wedge)
        # Quadrant Avant-Externe (Top-Left)
        wedge = patches.Wedge(center_L, radius_L, 90, 180, width=RING_WIDTH, 
                              facecolor=cmap(norm(data['L_Fore_Ext'])))
        ax.add_patch(wedge)
        # Quadrant Arrière-Externe (Bottom-Left)
        wedge = patches.Wedge(center_L, radius_L, 180, 270, width=RING_WIDTH, 
                              facecolor=cmap(norm(data['L_Hind_Ext'])))
        ax.add_patch(wedge)
        # Quadrant Arrière-Interne (Bottom-Right)
        wedge = patches.Wedge(center_L, radius_L, 270, 360, width=RING_WIDTH, 
                              facecolor=cmap(norm(data['L_Hind_Int'])))
        ax.add_patch(wedge)

        # --- DESSIN DU PIED DROIT ---
        # Quadrant Avant-Interne (Top-Right)
        wedge = patches.Wedge(center_R, radius_R, 0, 90, width=RING_WIDTH, 
                              facecolor=cmap(norm(data['R_Fore_Int'])))
        ax.add_patch(wedge)
        # Quadrant Avant-Externe (Top-Left)
        wedge = patches.Wedge(center_R, radius_R, 90, 180, width=RING_WIDTH, 
                              facecolor=cmap(norm(data['R_Fore_Ext'])))
        ax.add_patch(wedge)
        # Quadrant Arrière-Externe (Bottom-Left)
        wedge = patches.Wedge(center_R, radius_R, 180, 270, width=RING_WIDTH, 
                              facecolor=cmap(norm(data['R_Hind_Ext'])))
        ax.add_patch(wedge)
        # Quadrant Arrière-Interne (Bottom-Right)
        wedge = patches.Wedge(center_R, radius_R, 270, 360, width=RING_WIDTH, 
                              facecolor=cmap(norm(data['R_Hind_Int'])))
        ax.add_patch(wedge)

        # --- Ajout des Titres et Labels ---
        ax.text(-1.2, 1.3, 'PIED GAUCHE', ha='center', fontsize=12, fontweight='bold')
        ax.text(1.2, 1.3, 'PIED DROIT', ha='center', fontsize=12, fontweight='bold')
        fig.suptitle(f'Visualisation de la Pression - Temps : {data["time"]:.2f} s', 
                     fontsize=16, y=0.95)
        
        # Quadrants labels (simplifié)
        ax.text(center_L[0], center_L[1] + (MAX_RADIUS / 2), 'Avant', ha='center', va='center', fontsize=8, alpha=0.5)
        ax.text(center_L[0], center_L[1] - (MAX_RADIUS / 2), 'Arrière', ha='center', va='center', fontsize=8, alpha=0.5)
        ax.text(center_L[0] + (MAX_RADIUS / 2), center_L[1], 'Int', ha='center', va='center', fontsize=8, alpha=0.5)
        ax.text(center_L[0] - (MAX_RADIUS / 2), center_L[1], 'Ext', ha='center', va='center', fontsize=8, alpha=0.5)
        
        ax.text(center_R[0], center_R[1] + (MAX_RADIUS / 2), 'Avant', ha='center', va='center', fontsize=8, alpha=0.5)
        ax.text(center_R[0], center_R[1] - (MAX_RADIUS / 2), 'Arrière', ha='center', va='center', fontsize=8, alpha=0.5)
        ax.text(center_R[0] + (MAX_RADIUS / 2), center_R[1], 'Int', ha='center', va='center', fontsize=8, alpha=0.5)
        ax.text(center_R[0] - (MAX_RADIUS / 2), center_R[1], 'Ext', ha='center', va='center', fontsize=8, alpha=0.5)


    # --- Ajout de la Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Pression Agrégée (N/cm²)')

    # --- Lancement de l'animation ---
    print("Lancement de l'animation... (Cela peut prendre un moment)")
    # Intervalle est 100ms (0.1s), correspondant à la fréquence de 10Hz des données
    ani = animation.FuncAnimation(fig, animate, frames=len(df), interval=100, blit=False)
    
    plt.show()


# --- Exécution Principale ---

def main():
    """
    Fonction principale pour exécuter le script.
    """
    df_processed = load_and_process_data(DATA_FILE)
    
    if df_processed is not None:
        # Réduire la taille du dataset pour un test rapide (optionnel)
        # df_processed = df_processed.iloc[::10] # Prend 1 image sur 10
        
        create_animation(df_processed)
    else:
        print("Arrêt du script en raison d'une erreur de chargement des données.")

if __name__ == "__main__":
    main()
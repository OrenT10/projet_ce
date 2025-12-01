# Foot Pressure Tracker & Visualizer

## 1. Description du Projet

Ce logiciel est une solution d'analyse biomécanique avancée combinant la vision par ordinateur et la télémétrie de pression. Il permet de visualiser en temps réel, via une interface de Réalité Augmentée (RA), les forces exercées par un utilisateur sur des semelles connectées.

Le système fusionne deux flux de données :
* **Suivi Cinématique :** Détection de la posture et position des pieds via une caméra (Intelligence Artificielle MediaPipe).
* **Données Dynamiques :** Réception des niveaux de pression via un protocole réseau UDP.

L'objectif est de fournir une visualisation intuitive de l'appui au sol, stabilisée et respectant la perspective de l'utilisateur.

## 2. Fonctionnalités Principales

* **Fusion de Capteurs :** Synchronisation automatique entre le squelette visuel (caméra) et les données de force (semelles).
* **Visualisation "Sticky Ground" :** Algorithme de stabilisation ancrant les graphiques de pression au sol, indépendamment des mouvements de caméra ou de la levée des pieds.
* **Enregistrement de Session :** Sauvegarde automatique du flux vidéo analysé au format `.avi` pour archivage et relecture.
* **Modes de Fonctionnement :**
    * **Mode LIVE :** Acquisition de données réelles via réseau UDP.
    * **Mode SIMULATION :** Génération de données synthétiques pour démonstration sans matériel.

## 3. Prérequis Techniques

Pour assurer le bon fonctionnement du logiciel, la configuration suivante est requise :

**Matériel :**
* Ordinateur sous Windows 10/11.
* Webcam.
* Semelles connectées configurées sur le même réseau local.

**Logiciel :**
* Python 3.9 ou version supérieure.
* Bibliothèques tierces : OpenCV, MediaPipe, NumPy, Matplotlib.
* Modèle IA pour la visualisation : 
  Le modèle utilisé pour la détection des pieds et la pose des marqueurs indiquant les pressions est  "pose_landmarker_full.task", il est de performance moyenne. Cependant un autre modèle plus puissant ou plus léger de mediapipe peut aussi être utilisé au détriment de la qualité de détection des pieds par le programme ou des performances du progamme en temps réel. 
Voir https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

## 4. Installation

Suivez ces étapes pour installer l'environnement sur une nouvelle machine :

1.  **Installation de Python :** Assurez-vous que Python est installé et ajouté au PATH système.
2.  **Installation des dépendances :** Ouvrez l'invite de commande (cmd) dans le dossier du projet et exécutez la commande suivante :
    ```bash
    pip install opencv-python mediapipe numpy matplotlib
    ```
3.  **Vérification des fichiers :** Assurez-vous que le répertoire contient impérativement les éléments suivants :
    * `main.py` (Exécutable principal)
    * `pressure_manager.py` (Gestionnaire de flux UDP) 
    * `visual_engine.py` (Moteur de rendu graphique)
    * `pose_landmarker_full.task` (Modèle IA de détection)

## 5. Mode Live
  
Le logiciel agit comme un serveur d'écoute UDP. Une configuration réseau stricte est nécessaire pour la communication avec les semelles.
  ### Configuration Réseau
1.  **Adresse IP :** L'ordinateur exécutant le logiciel doit être connecté sur le même réseau que le logiciel OpenGo et disposer d'une IP fixe à configurer dans le code main.py, 
2. **Port d'écoute :** Le pare-feu Windows doit autoriser le trafic entrant sur le port UDP **5005** (port UDP configurable sur le logiciel OpenGO).

 ### Architectures réseau complete  
 
## 6. Guide d'Utilisation

### Lancement
Exécutez le script principal via l'invite de commande ou un IDE :
```bash
python main.py

```

## 7. Scripts supplémentaires

Le dépot git contient des scripts supplémentaires qui peuvent être utiles dans la réalisation de tests et pour ameliorer votre comprehension du fonctionnement de notre programme. 
 ### Le script File_to_Visu.py
  Il est à utiliser avec les fichiers text exportables du logiciel OpenGo après un enregistrement via les capteurs. Il faudra vous assurer que le nom du fichier est le même que celui configuré dans le script. 
### Le script "LivePressure_Visualizer.py" :
il permet de faire la visualisation en temps réel uniquement il faudra pour cela respecter la structure évoquée au point 5. (Mode Live) 

### Le script trackingScript2.py est un script de visualisation en temps réel mais avec une visualisation différente. 

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
* Webcam fonctionnelle connectée par USB.
* Semelles connectées configurées sur le même réseau local.

**Logiciel :**
* Python 3.9 ou version supérieure.
* Bibliothèques tierces : OpenCV, MediaPipe, NumPy, Matplotlib.

## 4. Installation

Suivez ces étapes pour installer l'environnement sur une nouvelle machine :

1.  **Installation de Python :** Assurez-vous que Python est installé et ajouté au PATH système.
2.  **Installation des dépendances :** Ouvrez l'invite de commande (cmd) dans le dossier du projet et exécutez la commande suivante :
    ```bash
    pip install opencv-python mediapipe numpy matplotlib
    ```
3.  **Vérification des fichiers :** Assurez-vous que le répertoire contient impérativement les éléments suivants :
    * `main.py` (Exécutable principal) [cite: 1]
    * `pressure_manager.py` (Gestionnaire de flux UDP) 
    * `visual_engine.py` (Moteur de rendu graphique) [cite: 3]
    * `pose_landmarker_full.task` (Modèle IA de détection) [cite: 2]

## 5. Configuration Réseau (Mode Live)

Le logiciel agit comme un serveur d'écoute UDP. Une configuration réseau stricte est nécessaire pour la communication avec les semelles.

1.  **Adresse IP :** L'ordinateur exécutant le logiciel doit disposer d'une IP fixe correspondant à la configuration des semelles.
    * [cite_start]IP Cible recommandée : `192.168.1.26` (Voir commentaire dans `main.py`) 
2.  [cite_start]**Port d'écoute :** Le pare-feu Windows doit autoriser le trafic entrant sur le port UDP **5005**[cite: 4].

## 6. Guide d'Utilisation

### Lancement
Exécutez le script principal via l'invite de commande ou un IDE :
```bash
python main.py

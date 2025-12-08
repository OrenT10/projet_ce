import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration du modèle MediaPipe ---
model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker_full.task')
if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Arquivo 'pose_landmarker_full.task' não encontrado.\n"
        "Baixe em: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=2,  # <<<<< Detecte deux personnes
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.PoseLandmarker.create_from_options(options)

# --- Paramètres de visualisation ---
BASE_RADIUS_PX = 30    # Rayon minimum (en pixels)
MAX_RADIUS_PX = 80     # Rayon maximum
MAX_FORCE_SIM = 300.0  # Force maximale simulée (N)

def get_color(value, max_val=1.0):
    """Mapeia valor [0, max_val] para cor BGR no estilo YlOrBr (amarelo -> laranja -> marrom)"""
    # HSV: amarelo = 60°, vermelho = 0°, marrom escuro ≈ vermelho escuro
    hue = int(60 * (1.0 - min(value / (max_val + 1e-6), 1.0)))
    sat, val = 255, 200  # valor ligeiramente reduzido para não ser muito claro
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, bgr))

def draw_foot_pressure(image, cx, cy, total_force, fore_int, fore_ext, hind_ext, hind_int, is_right=False):
    """Desenha os 4 setores de pressão de um pé usando cv2.ellipse"""
    radius = int(BASE_RADIUS_PX + (min(total_force, MAX_FORCE_SIM) / MAX_FORCE_SIM) * (MAX_RADIUS_PX - BASE_RADIUS_PX))
    max_zone = max(fore_int, fore_ext, hind_ext, hind_int, 1.0)

    # Definir zonas com ângulos (em graus)
    if not is_right:
        # Pé esquerdo: padrão
        zones = [
            (fore_int, -90,   0),   # Avant-Interne
            (fore_ext, -180, -90),  # Avant-Externe
            (hind_ext,  180,  90),  # Arrière-Externe
            (hind_int,   90,   0),  # Arrière-Interne
        ]
    else:
        # Pé direito: espelhado
        zones = [
            (fore_int,   90, 180),  # Avant-Interne
            (fore_ext,    0,  90),  # Avant-Externe
            (hind_ext,  -90,   0),  # Arrière-Externe
            (hind_int, -180, -90),  # Arrière-Interne
        ]

    for pressure, start_angle, end_angle in zones:
        color = get_color(pressure, max_zone)
        cv2.ellipse(
            image,
            (cx, cy),
            (radius, radius),
            0,
            start_angle,
            end_angle,
            color,
            thickness=-1  # preenchido
        )
        # Borda preta fina para separar os setores
        cv2.ellipse(
            image,
            (cx, cy),
            (radius, radius),
            0,
            start_angle,
            end_angle,
            (0, 0, 0),
            thickness=1
        )

def simulate_pressure():
    """Simula valores de pressão por zona (substitua por sensores reais depois)"""
    return {
        'fore_int': np.random.uniform(0, 150),
        'fore_ext': np.random.uniform(0, 150),
        'hind_ext': np.random.uniform(0, 100),
        'hind_int': np.random.uniform(0, 100),
    }

# --- Inicialização da webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_timestamp_ms = 0

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Processa a imagem com MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        annotated_image = image.copy()

        # Processa cada pessoa detectada (até 2)
        for pose_landmarks in detection_result.pose_landmarks:
            h, w = image.shape[:2]

            # Garantir que há landmarks suficientes
            if len(pose_landmarks) < 33:
                continue

            # Índices dos pés: 29=LEFT_HEEL, 31=LEFT_FOOT_INDEX, 30=RIGHT_HEEL, 32=RIGHT_FOOT_INDEX
            try:
                left_heel = pose_landmarks[29]
                left_toe = pose_landmarks[31]
                right_heel = pose_landmarks[30]
                right_toe = pose_landmarks[32]
            except IndexError:
                continue

            # Centro dos pés (média entre calcanhar e ponta)
            left_center = (
                int((left_heel.x + left_toe.x) * w / 2),
                int((left_heel.y + left_toe.y) * h / 2)
            )
            right_center = (
                int((right_heel.x + right_toe.x) * w / 2),
                int((right_heel.y + right_toe.y) * h / 2)
            )

            # Simular pressão
            left_press = simulate_pressure()
            right_press = simulate_pressure()
            total_left = sum(left_press.values())
            total_right = sum(right_press.values())

            # Desenhar nos pés
            draw_foot_pressure(
                annotated_image,
                left_center[0], left_center[1],
                total_left,
                left_press['fore_int'],
                left_press['fore_ext'],
                left_press['hind_ext'],
                left_press['hind_int'],
                is_right=False
            )
            draw_foot_pressure(
                annotated_image,
                right_center[0], right_center[1],
                total_right,
                right_press['fore_int'],
                right_press['fore_ext'],
                right_press['hind_ext'],
                right_press['hind_int'],
                is_right=True
            )

        # Mostrar resultado
        cv2.imshow('Pressão nos Pés - Até 2 Pessoas', annotated_image)
        frame_timestamp_ms += 1

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import requests
import time
from collections import deque

# Configuración
URL = "http://127.0.0.1:8000/predecir"
SECUENCIA_FRAMES = 60  # cantidad de frames a acumular
FEATURES_POR_FRAME = 147
VENTANA = deque(maxlen=SECUENCIA_FRAMES)
INTERVALO_SEGUNDOS = 0.3
last_prediction_time = 0

# Repeticiones para confirmar gesto
prediccion_anterior = None
contador_repeticiones = 0
FRAMES_REQUERIDOS = 3

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Puntos de pose
POSE_IDX = [
    mp_holistic.PoseLandmark.NOSE,
    mp_holistic.PoseLandmark.LEFT_SHOULDER,
    mp_holistic.PoseLandmark.RIGHT_SHOULDER,
    mp_holistic.PoseLandmark.LEFT_ELBOW,
    mp_holistic.PoseLandmark.RIGHT_ELBOW,
    mp_holistic.PoseLandmark.LEFT_WRIST,
    mp_holistic.PoseLandmark.RIGHT_WRIST
]

def extraer_landmarks(results):
    landmarks = []

    # Manos
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

    # Pose (7 puntos × 3)
    if results.pose_landmarks:
        for idx in POSE_IDX:
            lm = results.pose_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 21)

    return landmarks  # 147 features

print("📹 Cámara activada. Haz un gesto claro. Presiona ESC para salir.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    # Dibujar landmarks
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Extraer landmarks y guardar en ventana
    landmarks = extraer_landmarks(results)
    VENTANA.append(landmarks)

    # Mostrar cámara
    cv2.imshow("🧠 Reconocimiento en Tiempo Real", frame)
    if cv2.waitKey(1) == 27:
        break

    # Predicción cada INTERVALO
    if len(VENTANA) == SECUENCIA_FRAMES and time.time() - last_prediction_time >= INTERVALO_SEGUNDOS:
        last_prediction_time = time.time()

        secuencia = np.array(VENTANA)  # (60, 147)
        entrada = secuencia.flatten().tolist()  # (8820)

        try:
            response = requests.post(URL, json={"coordenadas": entrada})
            nueva_prediccion = response.json().get("palabra")

            # Confirmar por repeticiones consecutivas
            if nueva_prediccion == prediccion_anterior:
                contador_repeticiones += 1
            else:
                contador_repeticiones = 1
                prediccion_anterior = nueva_prediccion

            if contador_repeticiones == FRAMES_REQUERIDOS:
                print(f"✅ Gesto reconocido: {nueva_prediccion}")
                contador_repeticiones = 0

        except Exception as e:
            print("❌ Error al predecir:", e)

cap.release()
cv2.destroyAllWindows()

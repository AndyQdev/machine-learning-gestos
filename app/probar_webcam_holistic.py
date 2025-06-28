import cv2
import mediapipe as mp
import requests
import time

# Configuración
URL = "http://127.0.0.1:8000/predecir"
INTERVALO_SEGUNDOS = 0.3  # menor intervalo para más sensibilidad
FRAMES_REQUERIDOS = 3     # cuántas veces debe repetirse el mismo gesto
last_prediction_time = 0

# Variables de control
prediccion_anterior = None
contador_repeticiones = 0

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Índices relevantes
CARA_IDX = list(range(0, 30))
POSE_IDX = [
    mp_holistic.PoseLandmark.LEFT_SHOULDER,
    mp_holistic.PoseLandmark.RIGHT_SHOULDER,
    mp_holistic.PoseLandmark.LEFT_ELBOW,
    mp_holistic.PoseLandmark.RIGHT_ELBOW,
    mp_holistic.PoseLandmark.LEFT_WRIST,
    mp_holistic.PoseLandmark.RIGHT_WRIST,
    mp_holistic.PoseLandmark.NOSE,
    mp_holistic.PoseLandmark.LEFT_HIP,
    mp_holistic.PoseLandmark.RIGHT_HIP,
    mp_holistic.PoseLandmark.LEFT_EYE,
    mp_holistic.PoseLandmark.RIGHT_EYE,
    mp_holistic.PoseLandmark.MOUTH_RIGHT
]

cap = cv2.VideoCapture(0)
print("🖐 Reconocimiento robusto en tiempo real con Holistic. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    # Dibujar landmarks
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Mostrar cámara
    cv2.imshow("🧠 Lenguaje de Señas (Holistic Mejorado)", frame)
    if cv2.waitKey(1) == 27:
        break

    # Predicción con intervalo configurable
    if time.time() - last_prediction_time >= INTERVALO_SEGUNDOS:
        last_prediction_time = time.time()
        coords = []

        # Manos
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                for lm in hand.landmark:
                    coords.extend([lm.x, lm.y])
            else:
                coords.extend([0.0] * 42)

        # Cara
        if results.face_landmarks:
            for idx in CARA_IDX:
                lm = results.face_landmarks.landmark[idx]
                coords.extend([lm.x, lm.y])
        else:
            coords.extend([0.0] * len(CARA_IDX) * 2)

        # Pose
        if results.pose_landmarks:
            for idx in POSE_IDX:
                lm = results.pose_landmarks.landmark[idx]
                coords.extend([lm.x, lm.y])
        else:
            coords.extend([0.0] * len(POSE_IDX) * 2)

        # Validar y predecir
        if len(coords) == 168:
            try:
                response = requests.post(URL, json={"coordenadas": coords})
                nueva_prediccion = response.json().get("palabra")

                # Filtro por repetición y diferencia
                if nueva_prediccion == prediccion_anterior:
                    contador_repeticiones += 1
                else:
                    contador_repeticiones = 1  # reset
                    prediccion_anterior = nueva_prediccion

                if contador_repeticiones == FRAMES_REQUERIDOS:
                    print("✅ Gesto reconocido:", nueva_prediccion)
                    contador_repeticiones = 0  # para evitar repetir la misma predicción

            except Exception as e:
                print("❌ Error al enviar datos:", e)
        else:
            print("⚠ Coordenadas incompletas")

cap.release()
cv2.destroyAllWindows()

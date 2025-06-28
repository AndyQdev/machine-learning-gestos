import cv2
import mediapipe as mp
import os
import pandas as pd

DATASET = "dataset/"
OUTPUT_CSV = "data/dataset_holistic.csv"
filas = []

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

# Detectar etiquetas
PALABRAS = [nombre for nombre in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET, nombre))]
print(f"📁 Palabras detectadas: {PALABRAS}")

# Puntos clave que usaremos de la cara y del cuerpo (simplificado)
CARA_IDX = list(range(0, 30))  # puedes probar con más si tu PC lo soporta
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

for palabra in PALABRAS:
    carpeta = os.path.join(DATASET, palabra)
    for archivo in os.listdir(carpeta):
        video_path = os.path.join(carpeta, archivo)
        cap = cv2.VideoCapture(video_path)

        if cap.get(cv2.CAP_PROP_FRAME_COUNT) < 2:
            print(f"⚠ Video corrupto: {video_path}")
            cap.release()
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)

            fila = []

            # Manos
            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        fila.extend([lm.x, lm.y])
                else:
                    fila.extend([0.0] * 42)

            # Cara simplificada
            if results.face_landmarks:
                for idx in CARA_IDX:
                    lm = results.face_landmarks.landmark[idx]
                    fila.extend([lm.x, lm.y])
            else:
                fila.extend([0.0] * len(CARA_IDX) * 2)

            # Pose simplificada
            if results.pose_landmarks:
                for idx in POSE_IDX:
                    lm = results.pose_landmarks.landmark[idx]
                    fila.extend([lm.x, lm.y])
            else:
                fila.extend([0.0] * len(POSE_IDX) * 2)

            fila.append(palabra)
            filas.append(fila)

        cap.release()

# Guardar CSV
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(filas)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Dataset enriquecido guardado en: {OUTPUT_CSV}")

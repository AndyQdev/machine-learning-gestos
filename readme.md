# 🤟 App de Comunicación Inclusiva – Reconocimiento de Lenguaje de Señas

Este proyecto permite recolectar datos desde videos de gestos en lenguaje de señas, entrenar un modelo LSTM y realizar predicción en tiempo real a través de una API con FastAPI.

---

## 📁 Estructura del Proyecto

python app/recolectar_holistic.py
python app/entrenar_nn_holistic.py

uvicorn app.main_nn_holistic:app --reload --port 8000

en otra terminal:
python app/probar_webcam_holistic.py
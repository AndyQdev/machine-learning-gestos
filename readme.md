# 🤟 App de Comunicación Inclusiva – Reconocimiento de Lenguaje de Señas

Este proyecto permite recolectar datos desde videos de gestos en lenguaje de señas, entrenar un modelo LSTM y realizar predicción en tiempo real a través de una API con FastAPI.

---
instalar dependencias:
pip install -r requirements.txt
## 📁 Estructura del Proyecto

para entrenar:
python interactive_dataset_creator.py

python app/entrenar_nn_holistic.py

y luego correr
uvicorn app.main_nn_holistic:app --reload --port 8000

en otra terminal para probar gestos:
python app/probar_webcam_holistic.py
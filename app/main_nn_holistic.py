from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import os

# 🧠 Modelo MLP con entrada 8820 = 60 frames × 147 features
class SignClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 📂 Rutas absolutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
label_path = os.path.join(BASE_DIR, "modelo_nn", "label_encoder_holistic.txt")
model_path = os.path.join(BASE_DIR, "modelo_nn", "modelo_holistic.pt")

# 🔖 Cargar etiquetas
with open(label_path, "r") as f:
    etiquetas = [line.strip() for line in f.readlines()]

# ⚙️ Configuración del modelo
input_size = 60 * 147  # = 8820
hidden_size = 256
num_classes = len(etiquetas)

# 🔁 Cargar modelo
modelo = SignClassifier(input_size, hidden_size, num_classes)
modelo.load_state_dict(torch.load(model_path))
modelo.eval()

# 🚀 Inicializar FastAPI
app = FastAPI(title="API de Reconocimiento de Gestos", version="1.0")

# 🧾 Entrada esperada: lista de 8820 floats (una secuencia normalizada completa)
class Entrada(BaseModel):
    coordenadas: list[float] = Field(
        ..., 
        min_items=input_size, 
        max_items=input_size,
        description="8820 coordenadas (60 frames × 147 features por frame)"
    )

# 🧠 Endpoint de predicción
@app.post("/predecir")
def predecir(data: Entrada):
    if len(data.coordenadas) != input_size:
        raise HTTPException(status_code=400, detail=f"Se requieren exactamente {input_size} coordenadas")

    x = torch.tensor([data.coordenadas], dtype=torch.float32)
    with torch.no_grad():
        salida = modelo(x)
        indice = torch.argmax(salida, dim=1).item()
        palabra = etiquetas[indice]
        return {"palabra": palabra}

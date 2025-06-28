from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import os

# 🧠 Modelo MLP con entrada ajustada
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

# 📄 Cargar etiquetas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
label_path = os.path.join(BASE_DIR, "modelo_nn", "label_encoder_holistic.txt")
model_path = os.path.join(BASE_DIR, "modelo_nn", "modelo_holistic.pt")
with open(label_path, "r") as f:
    etiquetas = [line.strip() for line in f.readlines()]

# 🔁 Cargar modelo entrenado
input_size = 168
hidden_size = 256
num_classes = len(etiquetas)
modelo = SignClassifier(input_size, hidden_size, num_classes)
modelo.load_state_dict(torch.load(model_path))
modelo.eval()

# 🚀 FastAPI app
app = FastAPI()

class Entrada(BaseModel):
    coordenadas: list[float] = Field(..., min_items=input_size, max_items=input_size,
                                     description=f"{input_size} valores de coordenadas de MediaPipe Holistic")

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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

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

# Configurar templates y archivos estáticos
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 🧾 Entrada esperada: lista de 8820 floats (una secuencia normalizada completa)
class Entrada(BaseModel):
    coordenadas: list[float] = Field(
        ..., 
        min_items=input_size, 
        max_items=input_size,
        description="8820 coordenadas (60 frames × 147 features por frame)"
    )

# 📊 Modelos para el creador de dataset
class DatasetConfig(BaseModel):
    recordingDuration: int
    samplesPerWord: int
    totalSamples: int

class DatasetSample(BaseModel):
    word: str
    landmarks: list[float]
    timestamp: str

class DatasetRequest(BaseModel):
    words: list[str]
    samples: list[DatasetSample]
    config: DatasetConfig

# 🌐 Página principal con navegación
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# 🧠 Página de reconocimiento de gestos
@app.get("/reconocimiento", response_class=HTMLResponse)
async def reconocimiento(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 📝 Página de creación de dataset
@app.get("/crear-dataset", response_class=HTMLResponse)
async def crear_dataset(request: Request):
    return templates.TemplateResponse("dataset_creator.html", {"request": request})

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

# 📊 Endpoint para generar dataset
@app.post("/generar-dataset")
async def generar_dataset(data: DatasetRequest):
    try:
        # Crear directorio de datos si no existe
        os.makedirs("data", exist_ok=True)
        
        # Generar nombre único para el archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_web_{timestamp}.csv"
        filepath = os.path.join("data", filename)
        
        # Preparar datos para CSV
        csv_data = []
        
        for sample in data.samples:
            # Crear fila con landmarks + etiqueta
            row = sample.landmarks.copy()
            row.append(sample.word)
            csv_data.append(row)
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
        
        # Crear archivo de metadatos
        metadata = {
            "filename": filename,
            "words": data.words,
            "config": data.config.dict(),
            "created_at": datetime.now().isoformat(),
            "total_samples": len(data.samples),
            "samples_per_word": data.config.samplesPerWord
        }
        
        metadata_path = os.path.join("data", f"metadata_{timestamp}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "filename": filename,
            "total_samples": len(data.samples),
            "words": data.words,
            "message": f"Dataset generado con {len(data.samples)} muestras"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar dataset: {str(e)}")

# 📥 Endpoint para descargar dataset
@app.get("/descargar-dataset/{filename}")
async def descargar_dataset(filename: str):
    filepath = os.path.join("data", filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='text/csv'
    )

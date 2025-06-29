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
modelo_dir = os.path.join(BASE_DIR, "modelo_nn")

# 🔖 Variables globales para el modelo
modelo = None
etiquetas = []
input_size = 60 * 147  # = 8820
num_classes = 0

def cargar_modelo_default():
    """Carga el modelo por defecto si existe"""
    global modelo, etiquetas, num_classes
    
    # Buscar el modelo más reciente
    if os.path.exists(modelo_dir):
        modelos = []
        for item in os.listdir(modelo_dir):
            if item.startswith("modelo_"):
                timestamp = item.replace("modelo_", "")
                info_path = os.path.join(modelo_dir, item, f"info_{timestamp}.json")
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        modelos.append({
                            "timestamp": timestamp,
                            "created_at": info["created_at"]
                        })
                    except:
                        continue
        
        if modelos:
            # Ordenar por fecha y cargar el más reciente
            modelos.sort(key=lambda x: x["created_at"], reverse=True)
            timestamp = modelos[0]["timestamp"]
            
            info_path = os.path.join(modelo_dir, f"modelo_{timestamp}", f"info_{timestamp}.json")
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            # Cargar etiquetas
            with open(info["label_path"], "r", encoding='utf-8') as f:
                etiquetas = [line.strip() for line in f.readlines()]
            
            # Cargar modelo
            input_size = info["input_size"]
            num_classes = info["num_classes"]
            modelo = SignClassifier(input_size, 256, num_classes)
            modelo.load_state_dict(torch.load(info["modelo_path"]))
            modelo.eval()
            
            print(f"✅ Modelo cargado: {timestamp} ({len(etiquetas)} clases)")
            return True
    
    print("⚠️ No se encontró ningún modelo entrenado")
    return False

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
    sequenceFrames: int
    totalSamples: int

class DatasetSample(BaseModel):
    word: str
    landmarks: list[float]
    timestamp: str

class DatasetRequest(BaseModel):
    modelName: str
    words: list[str]
    samples: list[DatasetSample]
    config: DatasetConfig

class SelectModelRequest(BaseModel):
    timestamp: str

class TrainModelRequest(BaseModel):
    csv_filename: str = None

# 🌐 Página principal con navegación
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# 🧠 Página de reconocimiento de gestos
@app.get("/reconocimiento", response_class=HTMLResponse)
async def reconocimiento(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 📚 Página de aprendizaje de gestos
@app.get("/aprendizaje", response_class=HTMLResponse)
async def aprendizaje(request: Request):
    return templates.TemplateResponse("aprendizaje.html", {"request": request})

# 📝 Página de creación de dataset
@app.get("/crear-dataset", response_class=HTMLResponse)
async def crear_dataset(request: Request):
    return templates.TemplateResponse("dataset_creator.html", {"request": request})

# ⚙️ Página de gestión de modelos
@app.get("/gestionar-modelos", response_class=HTMLResponse)
async def gestionar_modelos(request: Request):
    return templates.TemplateResponse("model_manager.html", {"request": request})

# 🧠 Endpoint de predicción
@app.post("/predecir")
def predecir(data: Entrada):
    if modelo is None:
        raise HTTPException(status_code=500, detail="No hay modelo cargado")
    
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
        print(f"🔍 Datos recibidos:")
        print(f"   modelName: {data.modelName}")
        print(f"   words: {data.words}")
        print(f"   samples count: {len(data.samples)}")
        print(f"   config: {data.config}")
        print(f"   sequenceFrames: {data.config.sequenceFrames}")
        
        # Crear directorio de datos si no existe (en app/data desde la raíz)
        os.makedirs("app/data", exist_ok=True)
        
        # Usar el nombre del modelo para el archivo
        model_name_clean = data.modelName.replace(" ", "_").replace("/", "_").replace("\\", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{model_name_clean}_{timestamp}.csv"
        filepath = os.path.join("app/data", filename)
        
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
            "model_name": data.modelName,
            "words": data.words,
            "config": data.config.dict(),
            "created_at": datetime.now().isoformat(),
            "total_samples": len(data.samples),
            "samples_per_word": data.config.samplesPerWord,
            "sequence_frames": data.config.sequenceFrames,
            "features_per_frame": 147,
            "total_features": data.config.sequenceFrames * 147
        }
        
        metadata_path = os.path.join("app/data", f"metadata_{model_name_clean}_{timestamp}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 🚀 Entrenar modelo automáticamente
        print(f"📊 Dataset guardado en: {filepath}")
        print(f"📊 Configuración: {data.config.sequenceFrames} frames × 147 features = {data.config.sequenceFrames * 147} features totales")
        print(f"🚀 Iniciando entrenamiento automático...")
        training_result = await entrenar_modelo_automatico(filepath, model_name_clean)
        print(f"📊 Resultado del entrenamiento: {training_result}")
        
        return {
            "success": True,
            "filename": filename,
            "model_name": data.modelName,
            "total_samples": len(data.samples),
            "words": data.words,
            "sequence_frames": data.config.sequenceFrames,
            "message": f"Dataset generado con {len(data.samples)} muestras de {data.config.sequenceFrames} frames cada una",
            "training": training_result
        }
        
    except Exception as e:
        print(f"❌ Error en generar_dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al generar dataset: {str(e)}")

# 🧠 Función para entrenar modelo automáticamente
async def entrenar_modelo_automatico(csv_path: str, model_name: str):
    """Entrena automáticamente un modelo con el dataset generado usando el script entrenar_nn_holistic.py"""
    import subprocess
    import sys
    try:
        print(f"🚀 Iniciando entrenamiento automático con dataset: {csv_path}")
        print(f"📝 Nombre del modelo: {model_name}")
        
        # Verificar que el archivo existe
        if not os.path.exists(csv_path):
            print(f"❌ Error: No se encuentra el archivo {csv_path}")
            return {
                "success": False,
                "error": f"Archivo no encontrado: {csv_path}",
                "message": "Error: Dataset no encontrado"
            }
        
        # Verificar que el script existe
        script_path = "app/entrenar_nn_holistic.py"
        if not os.path.exists(script_path):
            print(f"❌ Error: No se encuentra el script {script_path}")
            return {
                "success": False,
                "error": f"Script no encontrado: {script_path}",
                "message": "Error: Script de entrenamiento no encontrado"
            }
        
        print(f"📁 Ejecutando: {sys.executable} {script_path} {csv_path} {model_name}")
        
        result = subprocess.run([sys.executable, script_path, csv_path, model_name], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        print(f"📊 Resultado del entrenamiento:")
        print(f"   Return code: {result.returncode}")
        print(f"   STDOUT: {result.stdout[:500]}...")  # Primeros 500 caracteres
        print(f"   STDERR: {result.stderr[:500]}...")  # Primeros 500 caracteres
        
        if result.returncode == 0:
            return {
                "success": True,
                "stdout": result.stdout,
                "message": "Modelo entrenado exitosamente"
            }
        else:
            return {
                "success": False,
                "error": result.stderr,
                "message": "Error en entrenamiento"
            }
    except Exception as e:
        print(f"❌ Excepción en entrenamiento automático: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Error en entrenamiento automático"
        }

# 🧠 Endpoint para entrenar modelo manualmente
@app.post("/entrenar-modelo")
async def entrenar_modelo_manual(request: TrainModelRequest):
    """Entrena un modelo con un dataset específico"""
    try:
        csv_filename = request.csv_filename
        
        if csv_filename is None:
            # Buscar el dataset más reciente
            data_dir = "app/data"
            if not os.path.exists(data_dir):
                raise HTTPException(status_code=404, detail="No hay datasets disponibles")
            
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if not csv_files:
                raise HTTPException(status_code=404, detail="No hay archivos CSV disponibles")
            
            # Ordenar por fecha de creación
            csv_files.sort(key=lambda x: os.path.getctime(os.path.join(data_dir, x)), reverse=True)
            csv_filename = csv_files[0]
        
        csv_path = os.path.join("app/data", csv_filename)
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"Archivo {csv_filename} no encontrado")
        
        # Extraer timestamp del nombre del archivo
        dataset_timestamp = csv_filename.replace("dataset_web_", "").replace(".csv", "")
        
        # Entrenar modelo
        training_result = await entrenar_modelo_automatico(csv_path, dataset_timestamp)
        
        if training_result["success"]:
            return {
                "success": True,
                "dataset": csv_filename,
                "model_timestamp": training_result["stdout"],
                "message": "Modelo entrenado exitosamente"
            }
        else:
            raise HTTPException(status_code=500, detail=training_result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")

# 📥 Endpoint para descargar dataset
@app.get("/descargar-dataset/{filename}")
async def descargar_dataset(filename: str):
    filepath = os.path.join("app/data", filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='text/csv'
    )

# 🎯 Endpoint para listar modelos disponibles
@app.get("/listar-modelos")
async def listar_modelos():
    """Lista todos los modelos entrenados disponibles"""
    try:
        if not os.path.exists(modelo_dir):
            return {"modelos": []}
        
        modelos = []
        for item in os.listdir(modelo_dir):
            if item.startswith("modelo_"):
                timestamp = item.replace("modelo_", "")
                info_path = os.path.join(modelo_dir, item, f"info_{timestamp}.json")
                
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        modelos.append({
                            "timestamp": timestamp,
                            "model_name": info.get("model_name", timestamp),  # Usar nombre del modelo si existe
                            "classes": info["classes"],
                            "total_samples": info["total_samples"],
                            "created_at": info["created_at"],
                            "modelo_path": info["modelo_path"],
                            "label_path": info["label_path"]
                        })
                    except:
                        continue
        
        # Ordenar por fecha de creación (más reciente primero)
        modelos.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"modelos": modelos}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando modelos: {str(e)}")

# 📊 Endpoint para listar datasets disponibles
@app.get("/listar-datasets")
async def listar_datasets():
    """Lista todos los datasets disponibles para entrenamiento"""
    try:
        data_dir = "app/data"
        if not os.path.exists(data_dir):
            return {"datasets": []}
        
        datasets = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv') and filename.startswith('dataset_'):
                filepath = os.path.join(data_dir, filename)
                
                # Obtener información del archivo
                try:
                    df = pd.read_csv(filepath)
                    total_samples = len(df)
                    
                    # Intentar leer metadatos si existen
                    metadata_path = filepath.replace('.csv', '.json').replace('dataset_', 'metadata_')
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    
                    datasets.append({
                        "filename": filename,
                        "total_samples": total_samples,
                        "created_at": metadata.get("created_at", ""),
                        "words": metadata.get("words", []),
                        "file_size": os.path.getsize(filepath)
                    })
                except Exception as e:
                    print(f"Error leyendo {filename}: {e}")
                    continue
        
        # Ordenar por fecha de creación (más reciente primero)
        datasets.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"datasets": datasets}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando datasets: {str(e)}")

# 🎯 Endpoint para seleccionar modelo activo
@app.post("/seleccionar-modelo")
async def seleccionar_modelo(request: SelectModelRequest):
    """Selecciona un modelo para usar en reconocimiento"""
    try:
        modelo_dir_path = os.path.join(modelo_dir, f"modelo_{request.timestamp}")
        info_path = os.path.join(modelo_dir_path, f"info_{request.timestamp}.json")
        
        if not os.path.exists(info_path):
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        # Cargar el modelo seleccionado globalmente
        global modelo, etiquetas, input_size, num_classes
        
        # Cargar etiquetas
        with open(info["label_path"], 'r', encoding='utf-8') as f:
            etiquetas = [line.strip() for line in f.readlines()]
        
        # Cargar modelo
        input_size = info["input_size"]
        num_classes = info["num_classes"]
        modelo = SignClassifier(input_size, 256, num_classes)
        modelo.load_state_dict(torch.load(info["modelo_path"]))
        modelo.eval()
        
        return {
            "success": True,
            "modelo": {
                "timestamp": request.timestamp,
                "model_name": info.get("model_name", request.timestamp),
                "classes": info["classes"],
                "total_samples": info["total_samples"]
            },
            "message": f"Modelo {info.get('model_name', request.timestamp)} cargado exitosamente"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error seleccionando modelo: {str(e)}")

# 🎯 Endpoint para obtener palabras de un modelo
@app.post("/obtener-palabras-modelo")
async def obtener_palabras_modelo(request: SelectModelRequest):
    """Obtiene las palabras disponibles en un modelo específico"""
    try:
        modelo_dir_path = os.path.join(modelo_dir, f"modelo_{request.timestamp}")
        info_path = os.path.join(modelo_dir_path, f"info_{request.timestamp}.json")
        
        if not os.path.exists(info_path):
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        return {
            "success": True,
            "words": info["classes"],
            "model_name": info.get("model_name", request.timestamp),
            "total_samples": info["total_samples"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo palabras del modelo: {str(e)}")

# 🚀 Cargar modelo al iniciar
cargar_modelo_default()

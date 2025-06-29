import pandas as pd
import numpy as np
import os
from datetime import datetime

def crear_dataset_prueba():
    """Crea un dataset de prueba para verificar el flujo"""
    
    # Crear directorio si no existe
    os.makedirs("app/data", exist_ok=True)
    
    # Generar datos de prueba
    num_muestras = 20
    num_features = 8820  # 60 frames × 147 features
    
    # Crear datos aleatorios
    X = np.random.randn(num_muestras, num_features).astype(np.float32)
    
    # Crear etiquetas (2 clases: "hola", "gracias")
    y = np.random.choice(["hola", "gracias"], num_muestras)
    
    # Crear DataFrame
    df = pd.DataFrame(X)
    df['label'] = y
    
    # Guardar dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset_web_{timestamp}.csv"
    filepath = os.path.join("app/data", filename)
    
    df.to_csv(filepath, index=False)
    
    # Crear metadatos
    metadata = {
        "filename": filename,
        "words": ["hola", "gracias"],
        "config": {
            "recordingDuration": 3,
            "samplesPerWord": 10,
            "totalSamples": num_muestras
        },
        "created_at": datetime.now().isoformat(),
        "total_samples": num_muestras,
        "samples_per_word": 10
    }
    
    metadata_path = os.path.join("app/data", f"metadata_{timestamp}.json")
    import json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset de prueba creado:")
    print(f"   📁 Archivo: {filepath}")
    print(f"   📁 Metadatos: {metadata_path}")
    print(f"   📊 Muestras: {num_muestras}")
    print(f"   📝 Clases: {list(set(y))}")
    
    return filepath, timestamp

if __name__ == "__main__":
    crear_dataset_prueba() 
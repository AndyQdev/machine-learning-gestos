import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
from datetime import datetime
import sys

def get_dataset_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        # Buscar el dataset más reciente en app/data/
        data_dir = "app/data"
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith('dataset_')]
        if not csv_files:
            print("No se encontraron archivos CSV de dataset en app/data/")
            sys.exit(1)
        csv_files.sort(key=lambda x: os.path.getctime(os.path.join(data_dir, x)), reverse=True)
        return os.path.join(data_dir, csv_files[0])

def get_model_name():
    if len(sys.argv) > 2:
        return sys.argv[2]
    else:
        # Usar timestamp como fallback
        return datetime.now().strftime("%Y%m%d_%H%M%S")

# Cargar dataset generado con MediaPipe Holistic
dataset_path = get_dataset_path()
df = pd.read_csv(dataset_path)

# Separar coordenadas y etiquetas
X = df.iloc[:, :-1].values.astype("float32")  # todas las columnas excepto la última
y = df.iloc[:, -1].values                     # última columna (la etiqueta)

# Codificar etiquetas a números
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividir datos (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Convertir a tensores
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Definir modelo MLP
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

# Ajustar input_size según número de columnas
input_size = X.shape[1]  # normalmente 8820 si usaste 60 frames × 147 features
hidden_size = 256        # puedes subirlo si tienes GPU
num_classes = len(le.classes_)

model = SignClassifier(input_size, hidden_size, num_classes)

# Configuración de entrenamiento
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 40

print("Entrenando modelo con MediaPipe Holistic...")

for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        acc = (outputs.argmax(1) == y_train).float().mean()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {acc.item():.4f}")

# Generar timestamp único
timestamp = get_model_name()

# Crear directorio del modelo
modelo_dir = f"app/modelo_nn/modelo_{timestamp}"
os.makedirs(modelo_dir, exist_ok=True)

# Guardar modelo y etiquetas
modelo_path = os.path.join(modelo_dir, f"modelo_{timestamp}.pt")
label_path = os.path.join(modelo_dir, f"label_encoder_{timestamp}.txt")

torch.save(model.state_dict(), modelo_path)
with open(label_path, "w", encoding='utf-8') as f:
    for label in le.classes_:
        f.write(label + "\n")

# Crear archivo de información del modelo
info = {
    "timestamp": timestamp,
    "model_name": timestamp,  # El nombre del modelo es el mismo que el timestamp
    "created_at": datetime.now().isoformat(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "num_classes": num_classes,
    "classes": le.classes_.tolist(),
    "total_samples": len(X),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "modelo_path": modelo_path,
    "label_path": label_path,
    "dataset_source": dataset_path,
    "training_config": {
        "epochs": epochs,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "test_split": 0.2
    }
}

info_path = os.path.join(modelo_dir, f"info_{timestamp}.json")
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"Modelo entrenado y guardado en: {modelo_dir}")
print(f"   Modelo: {modelo_path}")
print(f"   Etiquetas: {label_path}")
print(f"   Info: {info_path}")
print(f"   Clases: {list(le.classes_)}")
print(f"   Total muestras: {len(X)}")

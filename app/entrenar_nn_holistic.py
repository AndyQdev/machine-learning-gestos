import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Cargar dataset generado con MediaPipe Holistic
df = pd.read_csv("data/dataset_holistic.csv")

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
input_size = X.shape[1]  # normalmente 168 si usaste 84+60+24
hidden_size = 256        # puedes subirlo si tienes GPU
num_classes = len(le.classes_)

model = SignClassifier(input_size, hidden_size, num_classes)

# Configuración de entrenamiento
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 40

print("🚀 Entrenando modelo con MediaPipe Holistic...")

for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        acc = (outputs.argmax(1) == y_train).float().mean()
        print(f"🧠 Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {acc.item():.4f}")

# Guardar modelo y etiquetas
os.makedirs("modelo_nn", exist_ok=True)
torch.save(model.state_dict(), "modelo_nn/modelo_holistic.pt")
with open("modelo_nn/label_encoder_holistic.txt", "w") as f:
    for label in le.classes_:
        f.write(label + "\n")

print("✅ Modelo entrenado y guardado como 'modelo_nn/modelo_holistic.pt'")

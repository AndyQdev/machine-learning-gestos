# 🧠 Reconocimiento de Gestos en Tiempo Real

Una aplicación web moderna para el reconocimiento de lenguaje de señas usando MediaPipe Holistic y redes neuronales.

## ✨ Características

- 🌐 **Interfaz web moderna** con diseño responsivo
- 🎥 **Captura de video en tiempo real** desde la cámara
- 🧠 **Reconocimiento automático** de gestos usando IA
- 📊 **Estadísticas en tiempo real** (frames, predicciones, confianza)
- 🎨 **Diseño atractivo** con gradientes y animaciones
- 📱 **Compatible con móviles** y diferentes dispositivos

## 🚀 Instalación y Uso

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo (si no existe)
```bash
cd app
python entrenar_nn_holistic.py
```

### 3. Ejecutar la aplicación
```bash
uvicorn app.main_nn_holistic:app --reload --port 8000
```

### 4. Abrir en el navegador
Ve a `http://localhost:8000` para acceder a la interfaz web.

## 🎮 Cómo usar la interfaz

1. **Iniciar Cámara**: Haz clic en "🎥 Iniciar Cámara" para comenzar
2. **Esperar Captura**: La aplicación automáticamente captura 60 frames
3. **Ver Predicción**: El gesto reconocido aparecerá en pantalla
4. **Captura Manual**: Usa "📸 Capturar Gesto" para predicción manual
5. **Detener**: Usa "⏹️ Detener" para cerrar la cámara

## 📁 Estructura del Proyecto

```
HackatonOficial/
├── app/
│   ├── main_nn_holistic.py      # API FastAPI principal
│   ├── entrenar_nn_holistic.py  # Script de entrenamiento
│   ├── templates/
│   │   └── index.html           # Página web principal
│   ├── static/
│   │   └── script.js            # JavaScript para la interfaz
│   ├── modelo_nn/               # Modelos entrenados
│   └── data/                    # Datos de entrenamiento
├── requirements.txt
└── README.md
```

## 🔧 Configuración

### Parámetros del modelo
- **Secuencia**: 60 frames por predicción
- **Features**: 147 características por frame (manos + pose)
- **Entrada total**: 8820 valores (60 × 147)

### Personalización
Puedes modificar los estilos CSS en `app/templates/index.html` para cambiar la apariencia de la interfaz.

## 🛠️ Tecnologías Utilizadas

- **Backend**: FastAPI, PyTorch, MediaPipe
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **IA**: Redes neuronales MLP para clasificación
- **Video**: WebRTC para captura de cámara

## 📝 Notas

- La aplicación actualmente simula la extracción de landmarks para demostración
- Para uso real, integra MediaPipe.js en el frontend
- El modelo requiere datos de entrenamiento específicos
- Asegúrate de tener permisos de cámara en el navegador

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
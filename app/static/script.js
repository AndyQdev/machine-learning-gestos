// Configuración de MediaPipe y captura de video
class GestureRecognition {
    constructor() {
        this.video = document.getElementById('videoElement');
        this.canvas = document.getElementById('canvasOutput');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.isCapturing = false;
        this.frameCount = 0;
        this.predictionsCount = 0;
        this.currentSequence = [];
        this.lastPredictionTime = 0;
        this.predictionInterval = 300; // ms - igual que en el script original
        
        // FPS tracking
        this.fpsCounter = 0;
        this.lastFpsTime = Date.now();
        this.currentFps = 0;
        
        // Control de predicciones repetidas
        this.lastPrediction = null;
        this.predictionRepeatCount = 0;
        this.requiredRepeats = 2; // Necesita 2 predicciones iguales consecutivas
        this.lastDisplayedPrediction = null;
        
        // Configuración
        this.SEQUENCE_LENGTH = 60;
        this.FEATURES_PER_FRAME = 147;
        
        // Elementos del DOM
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.captureBtn = document.getElementById('captureBtn');
        this.status = document.getElementById('status');
        this.progressBar = document.getElementById('progressBar');
        this.predictionDisplay = document.getElementById('predictionDisplay');
        this.predictionText = document.getElementById('predictionText');
        this.confirmationIndicator = document.getElementById('confirmationIndicator');
        this.confirmationCount = document.getElementById('confirmationCount');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.captureBtn.addEventListener('click', () => this.manualCapture());
    }
    
    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                } 
            });
            this.video.srcObject = this.stream;
            
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.captureBtn.disabled = false;
            
            this.updateStatus('Cámara iniciada - Listo para capturar', 'ready');
            this.startCapture();
        } catch (error) {
            this.updateStatus('Error al acceder a la cámara: ' + error.message, 'error');
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.captureBtn.disabled = true;
        
        this.updateStatus('Cámara detenida', 'ready');
        this.stopCapture();
    }
    
    manualCapture() {
        if (this.currentSequence.length >= this.SEQUENCE_LENGTH) {
            this.makePrediction();
        } else {
            this.updateStatus('Necesitas más frames para hacer una predicción', 'error');
        }
    }
    
    updateStatus(message, type) {
        this.status.textContent = 'Estado: ' + message;
        this.status.className = 'status ' + type;
    }
    
    updateProgress(percentage) {
        this.progressBar.style.width = percentage + '%';
        this.progressBar.textContent = Math.round(percentage) + '%';
    }
    
    updateStats() {
        document.getElementById('framesCaptured').textContent = this.frameCount;
        document.getElementById('predictionsMade').textContent = this.predictionsCount;
        document.getElementById('currentFPS').textContent = this.currentFps;
    }
    
    calculateFPS() {
        this.fpsCounter++;
        const now = Date.now();
        
        if (now - this.lastFpsTime >= 1000) { // Actualizar FPS cada segundo
            this.currentFps = this.fpsCounter;
            this.fpsCounter = 0;
            this.lastFpsTime = now;
        }
    }
    
    startCapture() {
        this.isCapturing = true;
        this.captureFrame();
    }
    
    stopCapture() {
        this.isCapturing = false;
        this.currentSequence = [];
        this.updateProgress(0);
        this.currentFps = 0;
        this.fpsCounter = 0;
        this.resetPredictionTracking();
    }
    
    resetPredictionTracking() {
        this.lastPrediction = null;
        this.predictionRepeatCount = 0;
        this.lastDisplayedPrediction = null;
        this.predictionDisplay.style.display = 'none';
        this.confirmationIndicator.style.display = 'none';
        this.updateConfirmationDots(0);
    }
    
    updateConfirmationDots(count) {
        // Actualizar los puntos visuales
        for (let i = 1; i <= 2; i++) {
            const dot = document.getElementById(`dot${i}`);
            if (i <= count) {
                dot.classList.add('active');
            } else {
                dot.classList.remove('active');
            }
        }
        
        // Actualizar el contador
        this.confirmationCount.textContent = `${count}/${this.requiredRepeats}`;
    }
    
    captureFrame() {
        if (!this.isCapturing) return;
        
        // Calcular FPS
        this.calculateFPS();
        
        // Extraer landmarks del frame actual
        const landmarks = this.extractLandmarks();
        this.currentSequence.push(landmarks);
        this.frameCount++;
        
        // Mantener solo los últimos SEQUENCE_LENGTH frames
        if (this.currentSequence.length > this.SEQUENCE_LENGTH) {
            this.currentSequence.shift();
        }
        
        // Actualizar progreso
        const progress = (this.currentSequence.length / this.SEQUENCE_LENGTH) * 100;
        this.updateProgress(progress);
        
        // Hacer predicción automática cuando tengamos suficientes frames
        if (this.currentSequence.length === this.SEQUENCE_LENGTH) {
            const now = Date.now();
            // Respetar el intervalo de predicción como en el script original
            if (now - this.lastPredictionTime >= this.predictionInterval) {
                this.lastPredictionTime = now;
                this.makePrediction();
            }
        }
        
        this.updateStats();
        
        // Continuar capturando
        requestAnimationFrame(() => this.captureFrame());
    }
    
    extractLandmarks() {
        // En una implementación real, aquí usarías MediaPipe para extraer landmarks
        // Por ahora, simulamos la extracción
        return this.generateMockLandmarks();
    }
    
    generateMockLandmarks() {
        // Simular 147 features por frame (manos + pose)
        const landmarks = [];
        for (let i = 0; i < this.FEATURES_PER_FRAME; i++) {
            landmarks.push(Math.random() * 2 - 1); // Valores entre -1 y 1
        }
        return landmarks;
    }
    
    async makePrediction() {
        if (this.currentSequence.length !== this.SEQUENCE_LENGTH) {
            this.updateStatus('Secuencia incompleta', 'error');
            return;
        }
        
        this.updateStatus('Procesando predicción...', 'processing');
        
        try {
            // Aplanar la secuencia para enviar al servidor
            const flattenedSequence = this.currentSequence.flat();
            
            const response = await fetch('/predecir', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    coordenadas: flattenedSequence
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                this.predictionsCount++;
                
                // Manejar predicciones repetidas
                this.handlePredictionResult(result.palabra);
                
                this.updateStatus('Predicción procesada', 'ready');
                
                // Actualizar confianza (simulada)
                const confidence = Math.floor(Math.random() * 30) + 70; // 70-100%
                document.getElementById('confidence').textContent = confidence + '%';
                
                this.updateStats();
            } else {
                throw new Error('Error en la respuesta del servidor');
            }
        } catch (error) {
            this.updateStatus('Error al hacer predicción: ' + error.message, 'error');
        }
    }
    
    handlePredictionResult(prediction) {
        // Si es la misma predicción que la anterior, incrementar contador
        if (prediction === this.lastPrediction) {
            this.predictionRepeatCount++;
        } else {
            // Nueva predicción, resetear contador
            this.predictionRepeatCount = 1;
            this.lastPrediction = prediction;
        }
        
        // Mostrar el indicador de confirmación
        this.confirmationIndicator.style.display = 'block';
        this.updateConfirmationDots(this.predictionRepeatCount);
        
        // Solo mostrar la predicción si se repite el número requerido de veces
        // Y si es diferente a la última mostrada
        if (this.predictionRepeatCount >= this.requiredRepeats && 
            prediction !== this.lastDisplayedPrediction) {
            
            this.lastDisplayedPrediction = prediction;
            this.predictionText.textContent = prediction;
            this.predictionDisplay.style.display = 'block';
            
            this.updateStatus('Gesto confirmado: ' + prediction, 'ready');
            
            // Ocultar el indicador de confirmación después de mostrar la predicción
            setTimeout(() => {
                this.confirmationIndicator.style.display = 'none';
                this.resetPredictionTracking();
            }, 2000);
        }
    }
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    const gestureRecognition = new GestureRecognition();
}); 
// Sistema de aprendizaje con avatar instructor
class LearningSystem {
    constructor() {
        this.video = document.getElementById('videoElement');
        this.stream = null;
        this.isCapturing = false;
        this.isLearningActive = false;
        this.frameCount = 0;
        this.predictionsCount = 0;
        this.currentSequence = [];
        this.lastPredictionTime = 0;
        this.predictionInterval = 300;
        this.selectedWord = null;
        
        // FPS tracking
        this.fpsCounter = 0;
        this.lastFpsTime = Date.now();
        this.currentFps = 0;
        
        // Configuración
        this.SEQUENCE_LENGTH = 60;
        this.FEATURES_PER_FRAME = 147;
        
        // Sistema de confirmación para evitar falsos positivos
        this.consecutiveCorrectPredictions = 0;
        this.requiredConsecutiveCorrect = 2; // Necesita 2 predicciones correctas consecutivas
        this.minLearningTime = 3000; // Mínimo 3 segundos de aprendizaje
        this.learningStartTime = 0;
        this.lastPrediction = null;
        
        // Elementos del DOM
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.startLearningBtn = document.getElementById('startLearningBtn');
        this.stopLearningBtn = document.getElementById('stopLearningBtn');
        this.messagesArea = document.getElementById('messagesArea');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.avatarContainer = document.getElementById('avatarContainer');
        this.successModal = document.getElementById('successModal');
        this.modalMessage = document.getElementById('modalMessage');
        this.continueLearningBtn = document.getElementById('continueLearningBtn');
        this.modalOverlay = document.getElementById('modalOverlay');
        
        // Elementos del selector de modelo
        this.modelSelect = document.getElementById('modelSelect');
        this.modelInfo = document.getElementById('modelInfo');
        this.modelClasses = document.getElementById('modelClasses');
        this.modelSamples = document.getElementById('modelSamples');
        this.wordGrid = document.getElementById('wordGrid');
        
        this.initializeEventListeners();
        this.loadModels();
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.startLearningBtn.addEventListener('click', () => this.startLearning());
        this.stopLearningBtn.addEventListener('click', () => this.stopLearning());
        
        // Agregar listener para el selector de modelo
        this.modelSelect.addEventListener('change', () => this.onModelChange());
        
        // Event listeners para la modal
        this.continueLearningBtn.addEventListener('click', () => this.hideSuccessModal());
        this.modalOverlay.addEventListener('click', () => this.hideSuccessModal());
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
            
            this.updateStatus('Cámara iniciada - Listo para aprendizaje', 'success');
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
        this.stopLearning();
        
        this.updateStatus('Cámara detenida', 'processing');
    }
    
    startLearning() {
        if (!this.selectedWord) {
            this.updateStatus('Primero selecciona una palabra para aprender', 'error');
            return;
        }
        
        if (!this.stream) {
            this.updateStatus('Primero inicia la cámara', 'error');
            return;
        }
        
        this.isLearningActive = true;
        this.startLearningBtn.disabled = true;
        this.stopLearningBtn.disabled = false;
        
        // Resetear variables de confirmación
        this.consecutiveCorrectPredictions = 0;
        this.learningStartTime = Date.now();
        this.lastPrediction = null;
        
        // Mostrar indicadores de confirmación
        this.showConfirmationIndicators();
        
        this.updateStatus(`Aprendizaje iniciado para "${this.selectedWord}". Imita el gesto del avatar`, 'processing');
        this.startCapture();
    }
    
    stopLearning() {
        this.isLearningActive = false;
        this.startLearningBtn.disabled = false;
        this.stopLearningBtn.disabled = true;
        this.stopCapture();
        
        // Ocultar indicadores de confirmación
        this.hideConfirmationIndicators();
        
        this.updateStatus('Aprendizaje detenido. Selecciona otra palabra para continuar.', 'processing');
    }
    
    updateStatus(message, type) {
        this.messagesArea.innerHTML = `<div class="message-${type}">${message}</div>`;
    }
    
    updateProgress(percentage) {
        this.progressFill.style.width = percentage + '%';
        this.progressFill.textContent = Math.round(percentage) + '%';
        this.progressText.textContent = `${this.currentSequence.length} de ${this.SEQUENCE_LENGTH} frames`;
    }
    
    updateStats() {
        document.getElementById('framesCaptured').textContent = this.frameCount;
        document.getElementById('predictionsMade').textContent = this.predictionsCount;
        document.getElementById('currentFPS').textContent = this.currentFps;
    }
    
    calculateFPS() {
        this.fpsCounter++;
        const now = Date.now();
        
        if (now - this.lastFpsTime >= 1000) {
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
    }
    
    captureFrame() {
        if (!this.isCapturing) return;
        
        this.calculateFPS();
        
        const landmarks = this.extractLandmarks();
        this.currentSequence.push(landmarks);
        this.frameCount++;
        
        if (this.currentSequence.length > this.SEQUENCE_LENGTH) {
            this.currentSequence.shift();
        }
        
        const progress = (this.currentSequence.length / this.SEQUENCE_LENGTH) * 100;
        this.updateProgress(progress);
        
        if (this.currentSequence.length === this.SEQUENCE_LENGTH) {
            const now = Date.now();
            if (now - this.lastPredictionTime >= this.predictionInterval) {
                this.lastPredictionTime = now;
                this.makePrediction();
            }
        }
        
        this.updateStats();
        requestAnimationFrame(() => this.captureFrame());
    }
    
    extractLandmarks() {
        return this.generateMockLandmarks();
    }
    
    generateMockLandmarks() {
        const landmarks = [];
        for (let i = 0; i < this.FEATURES_PER_FRAME; i++) {
            landmarks.push(Math.random() * 2 - 1);
        }
        return landmarks;
    }
    
    async makePrediction() {
        if (this.currentSequence.length !== this.SEQUENCE_LENGTH) {
            return;
        }
        
        try {
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
                
                this.handleLearningPrediction(result.palabra);
                
                const confidence = Math.floor(Math.random() * 30) + 70;
                document.getElementById('confidence').textContent = confidence + '%';
                
                this.updateStats();
            }
        } catch (error) {
            this.updateStatus('Error al hacer predicción: ' + error.message, 'error');
        }
    }
    
    handleLearningPrediction(prediction) {
        const currentTime = Date.now();
        const learningDuration = currentTime - this.learningStartTime;
        
        // Verificar que haya pasado el tiempo mínimo de aprendizaje
        if (learningDuration < this.minLearningTime) {
            this.updateStatus(`Espera ${Math.ceil((this.minLearningTime - learningDuration) / 1000)}s más antes de intentar`, 'processing');
            return;
        }
        
        if (prediction === this.selectedWord) {
            // Incrementar contador de predicciones correctas consecutivas
            this.consecutiveCorrectPredictions++;
            
            // Actualizar indicadores de confirmación
            this.updateConfirmationIndicators();
            
            // Verificar si tenemos suficientes predicciones correctas consecutivas
            if (this.consecutiveCorrectPredictions >= this.requiredConsecutiveCorrect) {
                // Detener automáticamente el aprendizaje cuando se confirma el gesto correcto
                this.stopLearning();
                
                // Ocultar indicadores de confirmación
                this.hideConfirmationIndicators();
                
                // Mostrar mensaje de éxito fijo abajo
                this.showSuccessMessage();
                
                // Actualizar el estado en el área de mensajes
                this.updateStatus('¡Perfecto! Has replicado el gesto correctamente. ¡Bien hecho!', 'success');
            } else {
                // Solo actualizar el estado sin mostrar ningún mensaje
                this.updateStatus(`Practicando "${this.selectedWord}"...`, 'processing');
            }
        } else {
            // Resetear contador si la predicción es incorrecta, pero sin mostrar mensaje de error
            this.consecutiveCorrectPredictions = 0;
            this.updateConfirmationIndicators();
            // No mostrar feedback visual para errores, solo actualizar el estado
            this.updateStatus(`Practicando "${this.selectedWord}"...`, 'processing');
        }
        
        this.lastPrediction = prediction;
    }
    
    showFeedback(message, type) {
        this.messagesArea.textContent = message;
        this.messagesArea.className = 'status ' + type;
        
        // Solo mostrar por 2 segundos para el mensaje de espera
        setTimeout(() => {
            this.messagesArea.textContent = '';
        }, 2000);
    }
    
    showSuccessFeedback(message, type) {
        this.messagesArea.textContent = message;
        this.messagesArea.className = 'status ' + type;
        
        // El mensaje de éxito se mantiene visible por más tiempo
        setTimeout(() => {
            this.messagesArea.textContent = '';
        }, 5000);
    }
    
    showSuccessMessage() {
        this.modalMessage.textContent = `¡Has dominado el gesto "${this.selectedWord}" correctamente!`;
        this.successModal.style.display = 'block';
        
        // Actualizar el estado en el área de mensajes
        this.updateStatus('¡Perfecto! Has replicado el gesto correctamente. ¡Bien hecho!', 'success');
    }
    
    hideSuccessModal() {
        this.successModal.style.display = 'none';
        this.updateStatus('Selecciona otra palabra para continuar aprendiendo', 'processing');
        this.updateProgress(0);
    }
    
    showContinueOption() {
        // Este método ya no se usa, pero lo mantenemos por compatibilidad
        this.showSuccessMessage();
    }
    
    async loadModels() {
        try {
            const response = await fetch('/listar-modelos');
            const data = await response.json();
            
            this.modelSelect.innerHTML = '<option value="">Seleccionar modelo...</option>';
            
            if (data.modelos && data.modelos.length > 0) {
                data.modelos.forEach(modelo => {
                    const option = document.createElement('option');
                    option.value = modelo.timestamp;
                    option.textContent = `${modelo.model_name} (${modelo.classes.join(', ')})`;
                    this.modelSelect.appendChild(option);
                });
                
                if (data.modelos.length > 0) {
                    this.modelSelect.value = data.modelos[0].timestamp;
                    this.onModelChange();
                }
            } else {
                this.modelSelect.innerHTML = '<option value="">No hay modelos disponibles</option>';
                this.updateStatus('No hay modelos entrenados disponibles', 'error');
            }
        } catch (error) {
            console.error('Error cargando modelos:', error);
            this.modelSelect.innerHTML = '<option value="">Error cargando modelos</option>';
            this.updateStatus('Error cargando modelos', 'error');
        }
    }
    
    async onModelChange() {
        const selectedTimestamp = this.modelSelect.value;
        
        if (!selectedTimestamp) {
            this.modelInfo.style.display = 'none';
            this.updateStatus('No hay modelo seleccionado', 'error');
            return;
        }
        
        try {
            const response = await fetch('/seleccionar-modelo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ timestamp: selectedTimestamp })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.modelClasses.textContent = data.modelo.classes.join(', ');
                this.modelSamples.textContent = data.modelo.total_samples;
                this.modelInfo.style.display = 'block';
                
                this.updateStatus(`Modelo "${data.modelo.model_name || selectedTimestamp}" cargado. Selecciona una palabra para aprender.`, 'success');
                
                this.loadWordsForLearning();
            } else {
                this.updateStatus('Error cargando modelo: ' + data.detail, 'error');
            }
        } catch (error) {
            this.updateStatus('Error seleccionando modelo: ' + error.message, 'error');
        }
    }
    
    async loadWordsForLearning() {
        if (!this.modelSelect.value) {
            this.updateStatus('Primero selecciona un modelo', 'error');
            return;
        }
        
        try {
            const response = await fetch('/obtener-palabras-modelo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ timestamp: this.modelSelect.value })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayWords(data.words);
                this.updateStatus('Palabras cargadas. Selecciona una para aprender.', 'success');
            } else {
                this.updateStatus('Error cargando palabras del modelo', 'error');
            }
        } catch (error) {
            this.updateStatus('Error cargando palabras: ' + error.message, 'error');
        }
    }
    
    displayWords(words) {
        this.wordGrid.innerHTML = '';
        
        words.forEach(word => {
            const wordBtn = document.createElement('div');
            wordBtn.className = 'word-btn';
            wordBtn.textContent = word;
            wordBtn.onclick = () => this.selectWord(word);
            this.wordGrid.appendChild(wordBtn);
        });
    }
    
    selectWord(word) {
        document.querySelectorAll('.word-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        event.target.classList.add('active');
        this.selectedWord = word;
        
        this.showAvatarForWord(word);
        
        this.updateStatus(`Palabra seleccionada: "${word}". Haz clic en "Iniciar Aprendizaje"`, 'processing');
    }
    
    showAvatarForWord(word) {
        const videoHTML = this.generateVideoHTML(word);
        this.avatarContainer.innerHTML = videoHTML;
        
        // Iniciar reproducción del video
        this.startVideoPlayback(word);
    }
    
    generateVideoHTML(word) {
        // Intentar diferentes variaciones del nombre del archivo
        const videoVariations = [
            word.toLowerCase(),
            word.toLowerCase().replace(/\s+/g, ''),
            word.toLowerCase().replace(/\s+/g, '_'),
            word.toLowerCase().replace(/\s+/g, '-')
        ];
        
        // Crear múltiples sources para diferentes variaciones
        const sources = videoVariations.map(variation => 
            `<source src="/static/videos/${variation}.mp4" type="video/mp4">`
        ).join('');
        
        return `
            <video id="gestureVideo" autoplay loop muted playsinline 
                   onerror="this.parentElement.innerHTML='<div class=\\'no-video-animation\\'><div class=\\'sad-face\\'>😢</div><p>No hay video para \\'${word}\\'</p></div>'">
                ${sources}
                Tu navegador no soporta el elemento de video.
            </video>
        `;
    }
    
    startVideoPlayback(word) {
        // Asegurar que el video se reproduzca correctamente
        setTimeout(() => {
            const video = document.getElementById('gestureVideo');
            if (video) {
                // Intentar reproducir el video
                video.play().catch(error => {
                    console.log('Video autoplay failed:', error);
                    // Si falla el autoplay, mostrar un botón para reproducir manualmente
                    this.showPlayButton();
                });
                
                // Agregar event listeners para manejar errores
                video.addEventListener('error', () => {
                    console.log('Video error occurred');
                });
                
                video.addEventListener('loadeddata', () => {
                    console.log('Video loaded successfully');
                });
            }
        }, 100);
    }
    
    showPlayButton() {
        const avatarContainer = document.getElementById('avatarContainer');
        if (avatarContainer) {
            const playButton = document.createElement('button');
            playButton.className = 'video-play-button';
            playButton.innerHTML = '▶️ Reproducir Video';
            playButton.onclick = () => {
                const video = document.getElementById('gestureVideo');
                if (video) {
                    video.play();
                    playButton.style.display = 'none';
                }
            };
            avatarContainer.appendChild(playButton);
        }
    }
    
    showConfirmationIndicators() {
        const confirmationSection = document.getElementById('confirmationSection');
        const confirmationIndicators = document.getElementById('confirmationIndicators');
        
        confirmationIndicators.innerHTML = '';
        
        for (let i = 0; i < this.requiredConsecutiveCorrect; i++) {
            const indicator = document.createElement('div');
            indicator.className = 'confirmation-indicator';
            indicator.id = `confirmation-${i}`;
            confirmationIndicators.appendChild(indicator);
        }
        
        confirmationSection.style.display = 'block';
    }
    
    updateConfirmationIndicators() {
        for (let i = 0; i < this.requiredConsecutiveCorrect; i++) {
            const indicator = document.getElementById(`confirmation-${i}`);
            if (indicator) {
                if (i < this.consecutiveCorrectPredictions) {
                    indicator.className = 'confirmation-indicator completed';
                } else if (i === this.consecutiveCorrectPredictions) {
                    indicator.className = 'confirmation-indicator active';
                } else {
                    indicator.className = 'confirmation-indicator';
                }
            }
        }
    }
    
    hideConfirmationIndicators() {
        const confirmationSection = document.getElementById('confirmationSection');
        confirmationSection.style.display = 'none';
    }
}

// Variables globales
let learningSystem;

// Funciones globales para el HTML
function startLearning() {
    learningSystem.startLearning();
}

function stopLearning() {
    learningSystem.stopLearning();
}

function startCamera() {
    learningSystem.startCamera();
}

function stopCamera() {
    learningSystem.stopCamera();
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    learningSystem = new LearningSystem();
}); 
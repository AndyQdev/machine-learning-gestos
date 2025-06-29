// Creador de Dataset Interactivo
class DatasetCreator {
    constructor() {
        this.video = document.getElementById('videoElement');
        this.stream = null;
        this.isRecording = false;
        this.words = [];
        this.currentWordIndex = 0;
        this.currentSampleIndex = 0;
        this.recordingDuration = 2;
        this.samplesPerWord = 10;
        this.sequenceFrames = 60; // Nuevo: frames por secuencia
        this.dataset = [];
        this.recordingInterval = null;
        this.countdownInterval = null;
        this.landmarksSequence = [];
        this.recordingStartTime = null;
        
        // Elementos del DOM
        this.status = document.getElementById('status');
        this.currentWordDisplay = document.getElementById('currentWordDisplay');
        this.currentWordText = document.getElementById('currentWordText');
        this.recordingControls = document.getElementById('recordingControls');
        this.progressSection = document.getElementById('progressSection');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.wordsList = document.getElementById('wordsList');
        this.recordingCounter = document.getElementById('recordingCounter');
        this.recordingIndicator = document.getElementById('recordingIndicator');
        
        // Botones
        this.recordBtn = document.getElementById('recordBtn');
        this.nextWordBtn = document.getElementById('nextWordBtn');
        this.finishBtn = document.getElementById('finishBtn');
        
        // Estadísticas
        this.wordsCountEl = document.getElementById('wordsCount');
        this.totalSamplesEl = document.getElementById('totalSamples');
        this.recordedSamplesEl = document.getElementById('recordedSamples');
        this.currentProgressEl = document.getElementById('currentProgress');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Enter key para agregar palabras
        document.getElementById('newWord').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.addWord();
            }
        });
        
        // Controles por teclado
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && this.isRecordingActive()) {
                e.preventDefault();
                this.recordSample();
            } else if (e.key === ' ' && this.isRecordingActive()) {
                e.preventDefault();
                this.recordSample();
            } else if (e.key === 'n' && this.nextWordBtn.disabled === false) {
                e.preventDefault();
                this.nextWord();
            } else if (e.key === 'f' && this.finishBtn.disabled === false) {
                e.preventDefault();
                this.finishDataset();
            }
        });
    }
    
    isRecordingActive() {
        return this.stream && this.words.length > 0 && this.currentWordIndex < this.words.length;
    }
    
    addWord() {
        const input = document.getElementById('newWord');
        const word = input.value.trim().toLowerCase();
        
        if (word && !this.words.includes(word)) {
            this.words.push(word);
            this.updateWordsList();
            this.updateStats();
            input.value = '';
            this.updateStatus('ready', `Palabra "${word}" agregada`);
        } else if (this.words.includes(word)) {
            this.updateStatus('error', 'Esta palabra ya existe');
        }
    }
    
    removeWord(word) {
        this.words = this.words.filter(w => w !== word);
        this.updateWordsList();
        this.updateStats();
    }
    
    updateWordsList() {
        this.wordsList.innerHTML = '';
        
        if (this.words.length === 0) {
            this.wordsList.innerHTML = '<div style="color: #999; text-align: center; padding: 20px;">No hay palabras agregadas</div>';
            return;
        }
        
        this.words.forEach(word => {
            const wordItem = document.createElement('div');
            wordItem.className = 'word-item';
            wordItem.innerHTML = `
                <span>${word}</span>
                <button class="remove-word" onclick="datasetCreator.removeWord('${word}')">×</button>
            `;
            this.wordsList.appendChild(wordItem);
        });
    }
    
    updateStats() {
        this.wordsCountEl.textContent = this.words.length;
        this.totalSamplesEl.textContent = this.words.length * this.samplesPerWord;
        this.recordedSamplesEl.textContent = this.dataset.length;
        
        const progress = this.words.length > 0 ? (this.dataset.length / (this.words.length * this.samplesPerWord)) * 100 : 0;
        this.currentProgressEl.textContent = `${Math.round(progress)}%`;
    }
    
    async startDatasetCreation() {
        if (this.words.length === 0) {
            this.updateStatus('error', 'Agrega al menos una palabra antes de comenzar');
            return;
        }
        
        this.recordingDuration = parseInt(document.getElementById('recordingDuration').value);
        this.samplesPerWord = parseInt(document.getElementById('samplesPerWord').value);
        this.sequenceFrames = parseInt(document.getElementById('sequenceFrames').value); // Nuevo
        
        if (this.recordingDuration < 1 || this.recordingDuration > 5) {
            this.updateStatus('error', 'La duración debe estar entre 1 y 5 segundos');
            return;
        }
        
        if (this.samplesPerWord < 1 || this.samplesPerWord > 50) {
            this.updateStatus('error', 'Las muestras por palabra deben estar entre 1 y 50');
            return;
        }
        
        if (this.sequenceFrames < 30 || this.sequenceFrames > 120) {
            this.updateStatus('error', 'Los frames deben estar entre 30 y 120');
            return;
        }
        
        try {
            // Iniciar cámara
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                } 
            });
            this.video.srcObject = this.stream;
            
            // Inicializar variables
            this.currentWordIndex = 0;
            this.currentSampleIndex = 0;
            this.dataset = [];
            
            // Mostrar controles de grabación
            this.recordingControls.style.display = 'flex';
            this.progressSection.style.display = 'block';
            this.currentWordDisplay.style.display = 'block';
            
            // Actualizar interfaz
            this.updateCurrentWord();
            this.updateProgress();
            this.updateStats();
            this.updateStatus('ready', `Cámara iniciada. Configurado para ${this.sequenceFrames} frames. Presiona ENTER o ESPACIO para grabar`);
            
        } catch (error) {
            this.updateStatus('error', 'Error al acceder a la cámara: ' + error.message);
        }
    }
    
    updateCurrentWord() {
        if (this.currentWordIndex < this.words.length) {
            this.currentWordText.textContent = this.words[this.currentWordIndex];
        } else {
            this.currentWordText.textContent = 'Completado';
        }
    }
    
    updateProgress() {
        const totalSamples = this.words.length * this.samplesPerWord;
        const currentSamples = this.dataset.length;
        const progress = totalSamples > 0 ? (currentSamples / totalSamples) * 100 : 0;
        
        this.progressFill.style.width = `${progress}%`;
        this.progressFill.textContent = `${Math.round(progress)}%`;
        this.progressText.textContent = `${currentSamples} de ${totalSamples} muestras`;
    }
    
    async recordSample() {
        if (this.isRecording) return;
        
        this.isRecording = true;
        this.recordBtn.disabled = true;
        this.recordBtn.textContent = '⏺️ Grabando...';
        this.landmarksSequence = [];
        this.recordingStartTime = Date.now();
        
        // Mostrar contador y indicador
        this.recordingCounter.style.display = 'block';
        this.recordingIndicator.style.display = 'block';
        
        this.updateStatus('recording', `Grabando muestra ${this.currentSampleIndex + 1} de "${this.words[this.currentWordIndex]}"`);
        
        // Iniciar countdown
        let remainingTime = this.recordingDuration;
        this.recordingCounter.textContent = `${remainingTime.toFixed(1)}s`;
        
        this.countdownInterval = setInterval(() => {
            remainingTime -= 0.1;
            this.recordingCounter.textContent = `${remainingTime.toFixed(1)}s`;
            
            if (remainingTime <= 0) {
                clearInterval(this.countdownInterval);
                this.stopRecording();
            }
        }, 100);
        
        // Simular grabación de landmarks con la nueva configuración de frames
        this.simulateLandmarksRecording();
        
        // Esperar la duración especificada
        await new Promise(resolve => setTimeout(resolve, this.recordingDuration * 1000));
    }
    
    simulateLandmarksRecording() {
        // Simular grabación de landmarks durante la duración especificada
        const frameInterval = 1000 / 30; // 30 FPS de grabación
        const totalFrames = Math.floor(this.recordingDuration * 1000 / frameInterval);
        
        for (let i = 0; i < totalFrames; i++) {
            setTimeout(() => {
                if (this.isRecording) {
                    // Generar landmarks simulados (147 features)
                    const landmarks = [];
                    for (let j = 0; j < 147; j++) {
                        landmarks.push(Math.random() * 2 - 1); // Valores entre -1 y 1
                    }
                    this.landmarksSequence.push(landmarks);
                }
            }, i * frameInterval);
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        this.recordBtn.disabled = false;
        this.recordBtn.textContent = '🎥 Grabar Muestra';
        
        // Ocultar contador y indicador
        this.recordingCounter.style.display = 'none';
        this.recordingIndicator.style.display = 'none';
        
        // Procesar secuencia de landmarks
        if (this.landmarksSequence.length > 0) {
            // Normalizar secuencia al número de frames configurado
            const normalizedSequence = this.normalizeSequence(this.landmarksSequence, this.sequenceFrames);
            
            // Agregar muestra al dataset
            const sample = {
                word: this.words[this.currentWordIndex],
                landmarks: normalizedSequence.flat(),
                timestamp: new Date().toISOString()
            };
            
            this.dataset.push(sample);
            this.currentSampleIndex++;
            
            // Actualizar interfaz
            this.updateProgress();
            this.updateStats();
            
            // Verificar si completamos la palabra actual
            if (this.currentSampleIndex >= this.samplesPerWord) {
                this.nextWordBtn.disabled = false;
                this.updateStatus('ready', `Completadas ${this.samplesPerWord} muestras de "${this.words[this.currentWordIndex]}"`);
            } else {
                this.updateStatus('ready', `Muestra ${this.currentSampleIndex} de ${this.samplesPerWord} grabada`);
            }
            
            // Verificar si completamos todo el dataset
            if (this.currentWordIndex >= this.words.length - 1 && this.currentSampleIndex >= this.samplesPerWord) {
                this.finishBtn.disabled = false;
            }
        } else {
            this.updateStatus('error', 'No se detectaron landmarks durante la grabación');
        }
    }
    
    normalizeSequence(sequence, targetFrames) {
        // Normalizar secuencia al número de frames configurado
        const normalized = [];
        
        if (sequence.length === 0) {
            return Array(targetFrames).fill(Array(147).fill(0));
        }
        
        if (sequence.length === targetFrames) {
            return sequence;
        }
        
        // Interpolación para ajustar a targetFrames
        for (let i = 0; i < targetFrames; i++) {
            const index = (i / (targetFrames - 1)) * (sequence.length - 1);
            const lowIndex = Math.floor(index);
            const highIndex = Math.min(lowIndex + 1, sequence.length - 1);
            const fraction = index - lowIndex;
            
            const frame = [];
            for (let j = 0; j < 147; j++) {
                const low = sequence[lowIndex][j] || 0;
                const high = sequence[highIndex][j] || 0;
                frame.push(low + (high - low) * fraction);
            }
            normalized.push(frame);
        }
        
        return normalized;
    }
    
    nextWord() {
        if (this.currentWordIndex < this.words.length - 1) {
            this.currentWordIndex++;
            this.currentSampleIndex = 0;
            this.nextWordBtn.disabled = true;
            this.updateCurrentWord();
            this.updateStatus('ready', `Siguiente palabra: "${this.words[this.currentWordIndex]}"`);
        }
    }
    
    async finishDataset() {
        this.updateStatus('processing', 'Generando archivo del dataset...');
        
        try {
            // Obtener el nombre del modelo
            const modelName = document.getElementById('modelName').value.trim();
            if (!modelName) {
                this.updateStatus('error', 'Por favor ingresa un nombre para el modelo');
                return;
            }
            
            // Preparar datos para el servidor
            const datasetData = {
                modelName: modelName,
                words: this.words,
                samples: this.dataset,
                config: {
                    recordingDuration: this.recordingDuration,
                    samplesPerWord: this.samplesPerWord,
                    sequenceFrames: this.sequenceFrames, // Nuevo
                    totalSamples: this.dataset.length
                }
            };
            
            // Enviar al servidor para generar el archivo
            const response = await fetch('/generar-dataset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(datasetData)
            });
            
            if (response.ok) {
                const result = await response.json();
                this.updateStatus('ready', `✅ Dataset generado exitosamente: ${result.filename}`);
                
                // Mostrar enlace de descarga
                const downloadLink = document.createElement('a');
                downloadLink.href = `/descargar-dataset/${result.filename}`;
                downloadLink.download = result.filename;
                downloadLink.textContent = '📥 Descargar Dataset';
                downloadLink.className = 'button-primary';
                downloadLink.style.display = 'block';
                downloadLink.style.marginTop = '20px';
                downloadLink.style.textAlign = 'center';
                
                this.progressSection.appendChild(downloadLink);
                
            } else {
                throw new Error('Error en la respuesta del servidor');
            }
            
        } catch (error) {
            this.updateStatus('error', 'Error al generar el dataset: ' + error.message);
        }
    }
    
    updateStatus(type, message) {
        this.status.textContent = 'Estado: ' + message;
        this.status.className = 'status ' + type;
    }
}

// Variables globales para funciones
let datasetCreator;

// Funciones globales para el HTML
function addWord() {
    datasetCreator.addWord();
}

function startDatasetCreation() {
    datasetCreator.startDatasetCreation();
}

function recordSample() {
    datasetCreator.recordSample();
}

function nextWord() {
    datasetCreator.nextWord();
}

function finishDataset() {
    datasetCreator.finishDataset();
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    datasetCreator = new DatasetCreator();
}); 
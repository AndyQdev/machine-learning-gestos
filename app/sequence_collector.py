import cv2
import mediapipe as mp
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

class GestureSequenceCollector:
    def __init__(self, dataset_path="dataset/", output_path="data/"):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.sequence_length = 30  # 30 frames por gesto (ajustable)
        
        # Configurar MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Índices de landmarks más relevantes para gestos
        self.hand_indices = list(range(21))  # Todos los puntos de la mano
        self.pose_indices = [
            0,   # Nariz
            11, 12,  # Hombros
            13, 14,  # Codos
            15, 16,  # Muñecas
            17, 18, 19, 20, 21, 22,  # Manos (adicionales)
        ]
        
        self.sequences = []
        self.labels = []
        self.scaler = StandardScaler()
        
    def extract_landmarks(self, results):
        """Extrae landmarks relevantes de los resultados de MediaPipe"""
        landmarks = []
        
        # Mano izquierda (21 puntos × 3 coordenadas = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Mano derecha (21 puntos × 3 coordenadas = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Pose relevante (13 puntos × 3 coordenadas = 39)
        if results.pose_landmarks:
            for idx in self.pose_indices:
                if idx < len(results.pose_landmarks.landmark):
                    lm = results.pose_landmarks.landmark[idx]
                    landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    landmarks.extend([0.0, 0.0, 0.0])
        else:
            landmarks.extend([0.0] * 39)
        
        return np.array(landmarks)  # Total: 165 features
    
    def normalize_sequence(self, sequence):
        """Normaliza una secuencia de landmarks"""
        sequence = np.array(sequence)
        
        # Normalización temporal: redimensionar a sequence_length frames
        if len(sequence) != self.sequence_length:
            # Interpolación lineal para ajustar longitud
            indices_old = np.linspace(0, len(sequence) - 1, len(sequence))
            indices_new = np.linspace(0, len(sequence) - 1, self.sequence_length)
            
            normalized_sequence = []
            for i in range(sequence.shape[1]):  # Para cada feature
                feature_values = sequence[:, i]
                interpolated = np.interp(indices_new, indices_old, feature_values)
                normalized_sequence.append(interpolated)
            
            sequence = np.array(normalized_sequence).T
        
        return sequence
    
    def process_video(self, video_path, label):
        """Procesa un video y extrae la secuencia de landmarks"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error abriendo video: {video_path}")
            return None
        
        frames_data = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(img_rgb)
            
            # Extraer landmarks
            landmarks = self.extract_landmarks(results)
            frames_data.append(landmarks)
            frame_count += 1
        
        cap.release()
        
        if len(frames_data) < 5:  # Muy pocos frames
            print(f"⚠️ Video muy corto: {video_path} ({len(frames_data)} frames)")
            return None
        
        # Normalizar secuencia
        normalized_sequence = self.normalize_sequence(frames_data)
        
        return normalized_sequence
    
    def collect_all_sequences(self):
        """Recolecta todas las secuencias del dataset"""
        print("🔄 Recolectando secuencias de gestos...")
        
        # Detectar palabras
        palabras = [nombre for nombre in os.listdir(self.dataset_path) 
                   if os.path.isdir(os.path.join(self.dataset_path, nombre))]
        
        print(f"📁 Palabras detectadas: {palabras}")
        
        total_sequences = 0
        failed_videos = 0
        
        for palabra in palabras:
            carpeta = os.path.join(self.dataset_path, palabra)
            archivos = [f for f in os.listdir(carpeta) 
                       if f.endswith(('.mp4', '.avi', '.mov'))]
            
            print(f"\n📹 Procesando '{palabra}': {len(archivos)} videos")
            
            palabra_sequences = 0
            
            for archivo in archivos:
                video_path = os.path.join(carpeta, archivo)
                
                # Procesar video
                sequence = self.process_video(video_path, palabra)
                
                if sequence is not None:
                    self.sequences.append(sequence)
                    self.labels.append(palabra)
                    palabra_sequences += 1
                    total_sequences += 1
                    print(f"  ✅ {archivo}: {sequence.shape}")
                else:
                    failed_videos += 1
                    print(f"  ❌ {archivo}: Falló")
            
            print(f"  📊 '{palabra}': {palabra_sequences} secuencias exitosas")
        
        print(f"\n✅ Recolección completada:")
        print(f"   Total secuencias: {total_sequences}")
        print(f"   Videos fallidos: {failed_videos}")
        print(f"   Forma de secuencia: {self.sequences[0].shape if self.sequences else 'N/A'}")
        
        return total_sequences > 0
    
    def save_sequences(self):
        """Guarda las secuencias procesadas"""
        if not self.sequences:
            print("❌ No hay secuencias para guardar")
            return
        
        os.makedirs(self.output_path, exist_ok=True)
        
        # Convertir a arrays de numpy
        X = np.array(self.sequences)
        y = np.array(self.labels)
        
        # Normalizar features (opcional pero recomendado)
        print("🔄 Normalizando features...")
        
        # Reshape para normalizar: (samples, timesteps, features) -> (samples*timesteps, features)
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Ajustar y transformar
        X_normalized = self.scaler.fit_transform(X_reshaped)
        
        # Volver a la forma original
        X_normalized = X_normalized.reshape(original_shape)
        
        # Guardar datos
        data = {
            'X': X_normalized,
            'y': y,
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'feature_count': X.shape[-1],
            'palabras': list(set(y))
        }
        
        with open(os.path.join(self.output_path, 'gesture_sequences.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Secuencias guardadas en: {os.path.join(self.output_path, 'gesture_sequences.pkl')}")
        print(f"   Forma final: {X_normalized.shape}")
        print(f"   Palabras únicas: {len(set(y))}")
        
        # Guardar también estadísticas
        stats = {
            'total_sequences': len(X),
            'sequence_length': self.sequence_length,
            'feature_count': X.shape[-1],
            'palabras': {palabra: list(y).count(palabra) for palabra in set(y)},
            'shape': X.shape
        }
        
        with open(os.path.join(self.output_path, 'dataset_stats.pkl'), 'wb') as f:
            pickle.dump(stats, f)
        
        return stats
    
    def run_collection(self):
        """Ejecuta todo el proceso de recolección"""
        success = self.collect_all_sequences()
        
        if success:
            stats = self.save_sequences()
            return stats
        else:
            print("❌ No se pudieron recolectar secuencias")
            return None

# Función principal
def collect_gesture_sequences():
    collector = GestureSequenceCollector()
    stats = collector.run_collection()
    
    if stats:
        print("\n" + "="*50)
        print("📊 ESTADÍSTICAS FINALES:")
        print("="*50)
        for palabra, count in stats['palabras'].items():
            print(f"   {palabra}: {count} secuencias")
        print(f"   Forma de datos: {stats['shape']}")
        print(f"   Longitud de secuencia: {stats['sequence_length']} frames")
        print(f"   Features por frame: {stats['feature_count']}")
    
    return stats

if __name__ == "__main__":
    stats = collect_gesture_sequences()
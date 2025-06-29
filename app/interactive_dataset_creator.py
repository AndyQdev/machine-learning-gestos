import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import defaultdict
import threading
import queue

import pandas as pd

class InteractiveGestureRecorder:
    def __init__(self, output_path="../data/"):
        self.output_path = output_path
        self.sequence_length = 60  # 2 segundos a 30fps
        self.recording_duration = 3.0  # 3 segundos por gesto
        
        # Configuración de MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Palabras a recolectar
        self.palabras = ["hola", "gracias", "tu", "el", "yo"]
        self.samples_per_word = 20  # 20 muestras por palabra
        
        # Almacenamiento de datos
        self.sequences = defaultdict(list)
        self.current_sequence = []
        self.recording = False
        self.current_word = ""
        self.current_sample = 0
        
        # Control de interfaz
        self.frame_queue = queue.Queue(maxsize=2)
        self.status_text = "Presiona ESPACIO para comenzar"
        
    def extract_landmarks(self, results):
        """Extrae landmarks de MediaPipe con coordenadas relativas"""
        landmarks = []
        
        # Mano izquierda (21 puntos × 3 coordenadas)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Mano derecha (21 puntos × 3 coordenadas)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Pose clave (hombros, codos, muñecas, cara)
        pose_points = [
            mp.solutions.holistic.PoseLandmark.NOSE,
            mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.holistic.PoseLandmark.LEFT_ELBOW,
            mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW,
            mp.solutions.holistic.PoseLandmark.LEFT_WRIST,
            mp.solutions.holistic.PoseLandmark.RIGHT_WRIST,
        ]
        
        if results.pose_landmarks:
            for point in pose_points:
                lm = results.pose_landmarks.landmark[point.value]
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 21)  # 7 puntos × 3 coordenadas
        
        return np.array(landmarks)  # Total: 147 features
    
    def normalize_sequence(self, sequence):
        """Normaliza secuencia a longitud fija"""
        sequence = np.array(sequence)
        
        if len(sequence) == 0:
            return np.zeros((self.sequence_length, 147))
        
        # Interpolación para ajustar a sequence_length
        if len(sequence) != self.sequence_length:
            indices_old = np.linspace(0, len(sequence) - 1, len(sequence))
            indices_new = np.linspace(0, len(sequence) - 1, self.sequence_length)
            
            normalized = []
            for i in range(sequence.shape[1]):
                feature_values = sequence[:, i]
                interpolated = np.interp(indices_new, indices_old, feature_values)
                normalized.append(interpolated)
            
            sequence = np.array(normalized).T
        
        return sequence
    
    def draw_interface(self, frame):
        """Dibuja la interfaz de usuario en el frame"""
        h, w = frame.shape[:2]
        
        # Fondo semitransparente para texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Información principal
        cv2.putText(frame, f"CREADOR DE DATASET DE GESTOS", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.current_word:
            progress_text = f"Palabra: {self.current_word.upper()} ({self.current_sample + 1}/{self.samples_per_word})"
            cv2.putText(frame, progress_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Estado actual
        if self.recording:
            # Cuenta regresiva o grabando
            elapsed = time.time() - self.recording_start_time
            remaining = max(0, self.recording_duration - elapsed)
            
            if remaining > 0:
                status_color = (0, 255, 0) if remaining > 1 else (0, 165, 255)
                cv2.putText(frame, f"GRABANDO: {remaining:.1f}s", (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Círculo de grabación
                cv2.circle(frame, (w-50, 50), 15, (0, 0, 255), -1)
            else:
                cv2.putText(frame, "PROCESANDO...", (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, self.status_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instrucciones
        cv2.putText(frame, "ESPACIO: Grabar | Q: Salir | R: Reiniciar palabra | D: Borrar último", (20, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Progreso general
        total_collected = sum(len(sequences) for sequences in self.sequences.values())
        total_needed = len(self.palabras) * self.samples_per_word
        
        cv2.putText(frame, f"Progreso total: {total_collected}/{total_needed}", (w-200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def get_next_word(self):
        """Determina la siguiente palabra a grabar"""
        for palabra in self.palabras:
            if len(self.sequences[palabra]) < self.samples_per_word:
                return palabra
        return None
    
    def start_recording(self):
        """Inicia la grabación de un gesto"""
        self.current_word = self.get_next_word()
        
        if not self.current_word:
            self.status_text = "¡DATASET COMPLETO! Presiona Q para salir"
            return False
        
        self.current_sample = len(self.sequences[self.current_word])
        self.current_sequence = []
        self.recording = True
        self.recording_start_time = time.time()
        
        print(f"\n🎬 Grabando '{self.current_word}' - Muestra {self.current_sample + 1}/{self.samples_per_word}")
        print(f"   Haz el gesto para '{self.current_word}' durante {self.recording_duration} segundos...")
        
        return True
    
    def stop_recording(self):
        """Detiene la grabación y procesa la secuencia"""
        if not self.recording or not self.current_sequence:
            return
        
        self.recording = False
        
        # Normalizar y guardar secuencia
        normalized_sequence = self.normalize_sequence(self.current_sequence)
        self.sequences[self.current_word].append(normalized_sequence)
        
        remaining = self.samples_per_word - len(self.sequences[self.current_word])
        
        print(f"✅ Gesto '{self.current_word}' guardado. Faltan {remaining} muestras de esta palabra.")
        
        if remaining == 0:
            print(f"🎉 ¡Palabra '{self.current_word}' completada!")
        
        # Actualizar estado
        next_word = self.get_next_word()
        if next_word:
            self.status_text = f"Siguiente: '{next_word}' - Presiona ESPACIO"
        else:
            self.status_text = "¡DATASET COMPLETO! Presiona S para guardar"
        
    def save_dataset(self):
        """Guarda el dataset recolectado"""
        if not any(self.sequences.values()):
            print("❌ No hay datos para guardar")
            return False
        
        os.makedirs(self.output_path, exist_ok=True)
        
        # Preparar datos
        all_sequences = []
        all_labels = []
        
        for palabra, sequences in self.sequences.items():
            for sequence in sequences:
                all_sequences.append(sequence)
                all_labels.append(palabra)
        
        X = np.array(all_sequences)
        y = np.array(all_labels)
        
        # Crear dataset
        dataset = {
            'X': X,
            'y': y,
            'sequence_length': self.sequence_length,
            'feature_count': X.shape[-1],
            'palabras': self.palabras,
            'samples_per_word': {palabra: len(sequences) for palabra, sequences in self.sequences.items()},
            'total_samples': len(X),
            'recording_duration': self.recording_duration
        }
        
        # Guardar
        dataset_path = os.path.join(self.output_path, 'interactive_gesture_dataset.pkl')
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n💾 Dataset guardado en: {dataset_path}")
        print(f"   📊 Total de muestras: {len(X)}")
        print(f"   📏 Forma de datos: {X.shape}")
        print(f"   📝 Palabras: {list(self.sequences.keys())}")
        
        # Mostrar distribución
        print("\n📈 Distribución de muestras:")
        for palabra in self.palabras:
            count = len(self.sequences[palabra])
            print(f"   {palabra}: {count}/{self.samples_per_word} muestras")
        # También guardar como CSV (formato compatible con entrenar_nn_holistic.py)
        csv_data = []

        for i in range(len(X)):
            row = X[i].flatten().tolist()  # Convierte de (60, 147) → 8820 columnas
            row.append(y[i])
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.output_path, "dataset_holistic.csv")
        df.to_csv(csv_path, index=False)
        print(f"📝 CSV generado para entrenamiento: {csv_path}")
        return True
    
    def run_collection(self):
        """Ejecuta el proceso de recolección interactiva"""
        print("🚀 Iniciando recolección interactiva de gestos")
        print(f"📝 Palabras a recolectar: {', '.join(self.palabras)}")
        print(f"🎯 {self.samples_per_word} muestras por palabra")
        print(f"⏱️  {self.recording_duration} segundos por gesto")
        print("\n" + "="*50)
        print("INSTRUCCIONES:")
        print("• Colócate frente a la cámara con buena iluminación")
        print("• Presiona ESPACIO para grabar cada gesto")
        print("• Haz el gesto claramente durante la grabación")
        print("• Presiona Q para salir en cualquier momento")
        print("• Presiona S para guardar el dataset")
        print("="*50)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: No se puede acceder a la cámara")
            return False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar con MediaPipe
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(img_rgb)
                
                # Dibujar landmarks
                self.mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                
                # Grabar si está en modo grabación
                if self.recording:
                    elapsed = time.time() - self.recording_start_time
                    
                    if elapsed < self.recording_duration:
                        # Extraer landmarks y agregar a secuencia
                        landmarks = self.extract_landmarks(results)
                        self.current_sequence.append(landmarks)
                    else:
                        # Terminar grabación
                        self.stop_recording()
                
                # Dibujar interfaz
                frame = self.draw_interface(frame)
                
                # Mostrar frame
                cv2.imshow('Creador de Dataset de Gestos', frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' ') and not self.recording:
                    self.start_recording()
                elif key == ord('s'):
                    if self.save_dataset():
                        print("✅ Dataset guardado exitosamente")
                    break
                elif key == ord('r'):
                    # Reiniciar palabra actual
                    if self.current_word:
                        self.sequences[self.current_word] = []
                        print(f"🔄 Reiniciada la palabra '{self.current_word}'")
                        self.status_text = f"Reiniciado '{self.current_word}' - Presiona ESPACIO"
                elif key == ord('d'):
                    # Borrar la última muestra de la palabra actual
                    if self.current_word and self.sequences[self.current_word]:
                        self.sequences[self.current_word].pop()
                        print(f"🗑️ Eliminada la última muestra de '{self.current_word}'")
                        self.status_text = f"Última muestra de '{self.current_word}' eliminada. Presiona ESPACIO"
                    else:
                        print("⚠️ No hay muestras para eliminar.")
        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido por el usuario")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
        
        return True

def create_interactive_dataset():
    """Función principal para crear dataset interactivo"""
    recorder = InteractiveGestureRecorder()
    success = recorder.run_collection()
    
    if success:
        print("\n🎉 ¡Recolección completada!")
        
        # Ofrecer guardar si no se guardó ya
        if any(recorder.sequences.values()):
            save_now = input("\n¿Quieres guardar el dataset ahora? (s/n): ").lower()
            if save_now == 's':
                recorder.save_dataset()
    
    return success

if __name__ == "__main__":
    create_interactive_dataset()
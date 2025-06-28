import cv2
import mediapipe as mp
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

class DatasetAnalyzer:
    def __init__(self, dataset_path="dataset/"):
        self.dataset_path = dataset_path
        self.stats = defaultdict(list)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False)
        
    def analyze_videos(self):
        """Analiza todos los videos del dataset"""
        print("🔍 Analizando dataset de videos...")
        
        palabras = [nombre for nombre in os.listdir(self.dataset_path) 
                   if os.path.isdir(os.path.join(self.dataset_path, nombre))]
        
        total_videos = 0
        video_details = []
        
        for palabra in palabras:
            carpeta = os.path.join(self.dataset_path, palabra)
            archivos = [f for f in os.listdir(carpeta) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            print(f"\n📁 Analizando '{palabra}': {len(archivos)} videos")
            
            for archivo in archivos:
                video_path = os.path.join(carpeta, archivo)
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    print(f"❌ No se puede abrir: {video_path}")
                    continue
                
                # Estadísticas del video
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Analizar calidad de landmarks
                valid_frames = 0
                hand_detection_count = 0
                pose_detection_count = 0
                face_detection_count = 0
                
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Procesar cada 5 frames para eficiencia
                    if frame_idx % 5 == 0:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.holistic.process(img_rgb)
                        
                        valid_frames += 1
                        
                        if results.left_hand_landmarks or results.right_hand_landmarks:
                            hand_detection_count += 1
                        if results.pose_landmarks:
                            pose_detection_count += 1
                        if results.face_landmarks:
                            face_detection_count += 1
                    
                    frame_idx += 1
                
                cap.release()
                
                # Calcular porcentajes de detección
                hand_detection_rate = (hand_detection_count / valid_frames * 100) if valid_frames > 0 else 0
                pose_detection_rate = (pose_detection_count / valid_frames * 100) if valid_frames > 0 else 0
                face_detection_rate = (face_detection_count / valid_frames * 100) if valid_frames > 0 else 0
                
                video_info = {
                    'palabra': palabra,
                    'archivo': archivo,
                    'frames': frame_count,
                    'fps': fps,
                    'duracion_seg': duration,
                    'resolucion': f"{width}x{height}",
                    'deteccion_manos_%': hand_detection_rate,
                    'deteccion_pose_%': pose_detection_rate,
                    'deteccion_cara_%': face_detection_rate,
                    'calidad_general': (hand_detection_rate + pose_detection_rate + face_detection_rate) / 3
                }
                
                video_details.append(video_info)
                self.stats[palabra].append(video_info)
                total_videos += 1
                
                print(f"  📹 {archivo}: {frame_count} frames, {duration:.1f}s, "
                      f"Manos:{hand_detection_rate:.1f}%, Pose:{pose_detection_rate:.1f}%, "
                      f"Cara:{face_detection_rate:.1f}%")
        
        print(f"\n✅ Análisis completado: {total_videos} videos analizados")
        return video_details
    
    def create_balance_report(self, video_details):
        """Crea reporte de balance del dataset"""
        df = pd.DataFrame(video_details)
        
        print("\n" + "="*60)
        print("📊 REPORTE DE BALANCE DEL DATASET")
        print("="*60)
        
        # 1. Distribución por palabra
        print("\n1️⃣ DISTRIBUCIÓN DE VIDEOS POR PALABRA:")
        palabra_counts = df['palabra'].value_counts()
        for palabra, count in palabra_counts.items():
            print(f"   {palabra}: {count} videos")
        
        # 2. Estadísticas de duración
        print("\n2️⃣ ESTADÍSTICAS DE DURACIÓN:")
        print(f"   Duración promedio: {df['duracion_seg'].mean():.2f} segundos")
        print(f"   Duración mínima: {df['duracion_seg'].min():.2f} segundos")
        print(f"   Duración máxima: {df['duracion_seg'].max():.2f} segundos")
        print(f"   Desviación estándar: {df['duracion_seg'].std():.2f} segundos")
        
        # 3. Calidad de detección
        print("\n3️⃣ CALIDAD DE DETECCIÓN PROMEDIO:")
        print(f"   Detección de manos: {df['deteccion_manos_%'].mean():.1f}%")
        print(f"   Detección de pose: {df['deteccion_pose_%'].mean():.1f}%")
        print(f"   Detección de cara: {df['deteccion_cara_%'].mean():.1f}%")
        print(f"   Calidad general: {df['calidad_general'].mean():.1f}%")
        
        # 4. Videos problemáticos
        print("\n4️⃣ VIDEOS CON BAJA CALIDAD DE DETECCIÓN (<70%):")
        problemas = df[df['calidad_general'] < 70]
        if len(problemas) > 0:
            for _, row in problemas.iterrows():
                print(f"   ⚠️ {row['palabra']}/{row['archivo']}: {row['calidad_general']:.1f}%")
        else:
            print("   ✅ Todos los videos tienen buena calidad de detección")
        
        # 5. Recomendaciones de balance
        print("\n5️⃣ RECOMENDACIONES DE BALANCE:")
        min_videos = palabra_counts.min()
        max_videos = palabra_counts.max()
        
        if max_videos / min_videos > 2:
            print("   ⚠️ Dataset desbalanceado detectado!")
            print(f"   La palabra '{palabra_counts.idxmax()}' tiene {max_videos} videos")
            print(f"   La palabra '{palabra_counts.idxmin()}' tiene {min_videos} videos")
            print("   Sugerencias:")
            print("   - Grabar más videos para palabras con pocos ejemplos")
            print("   - O usar data augmentation")
            print("   - O reducir videos de palabras sobre-representadas")
        else:
            print("   ✅ Dataset bien balanceado")
        
        return df
    
    def create_visualizations(self, df):
        """Crea visualizaciones del análisis"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis del Dataset de Gestos', fontsize=16, fontweight='bold')
        
        # 1. Distribución de videos por palabra
        palabra_counts = df['palabra'].value_counts()
        axes[0,0].bar(palabra_counts.index, palabra_counts.values, color='skyblue')
        axes[0,0].set_title('Videos por Palabra')
        axes[0,0].set_xlabel('Palabra')
        axes[0,0].set_ylabel('Número de Videos')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Distribución de duraciones
        axes[0,1].hist(df['duracion_seg'], bins=20, color='lightgreen', alpha=0.7)
        axes[0,1].set_title('Distribución de Duraciones')
        axes[0,1].set_xlabel('Duración (segundos)')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].axvline(df['duracion_seg'].mean(), color='red', linestyle='--', 
                         label=f'Media: {df["duracion_seg"].mean():.2f}s')
        axes[0,1].legend()
        
        # 3. Calidad de detección por palabra
        quality_by_word = df.groupby('palabra')['calidad_general'].mean().sort_values()
        axes[1,0].barh(quality_by_word.index, quality_by_word.values, color='orange')
        axes[1,0].set_title('Calidad de Detección por Palabra')
        axes[1,0].set_xlabel('Calidad Promedio (%)')
        axes[1,0].axvline(70, color='red', linestyle='--', label='Umbral mínimo')
        axes[1,0].legend()
        
        # 4. Correlación duración vs calidad
        axes[1,1].scatter(df['duracion_seg'], df['calidad_general'], alpha=0.6, color='purple')
        axes[1,1].set_title('Duración vs Calidad de Detección')
        axes[1,1].set_xlabel('Duración (segundos)')
        axes[1,1].set_ylabel('Calidad General (%)')
        
        # Ajustar correlación
        correlation = df['duracion_seg'].corr(df['calidad_general'])
        axes[1,1].text(0.05, 0.95, f'Correlación: {correlation:.3f}', 
                      transform=axes[1,1].transAxes, fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Visualizaciones guardadas en 'dataset_analysis.png'")
    
    def suggest_improvements(self, df):
        """Sugiere mejoras específicas"""
        print("\n" + "="*60)
        print("💡 SUGERENCIAS DE MEJORA")
        print("="*60)
        
        # Identificar problemas específicos
        issues = []
        
        # 1. Videos muy cortos o muy largos
        short_videos = df[df['duracion_seg'] < 1.0]
        long_videos = df[df['duracion_seg'] > 5.0]
        
        if len(short_videos) > 0:
            issues.append(f"📹 {len(short_videos)} videos muy cortos (<1s)")
        if len(long_videos) > 0:
            issues.append(f"📹 {len(long_videos)} videos muy largos (>5s)")
        
        # 2. Baja detección de manos (crítico para gestos)
        low_hand_detection = df[df['deteccion_manos_%'] < 80]
        if len(low_hand_detection) > 0:
            issues.append(f"👋 {len(low_hand_detection)} videos con baja detección de manos")
        
        # 3. Dataset desbalanceado
        palabra_counts = df['palabra'].value_counts()
        if palabra_counts.max() / palabra_counts.min() > 2:
            issues.append("⚖️ Dataset desbalanceado")
        
        if issues:
            print("🔍 PROBLEMAS DETECTADOS:")
            for issue in issues:
                print(f"   {issue}")
        
        print("\n🛠️ ACCIONES RECOMENDADAS:")
        print("1. Normalizar duración de videos a 2-4 segundos")
        print("2. Asegurar buena iluminación y contraste de manos")
        print("3. Grabar gestos con movimientos claros y completos")
        print("4. Balancear número de videos por palabra")
        print("5. Implementar arquitectura secuencial (LSTM/GRU)")
        
        return issues

# Función principal de análisis
def run_analysis():
    analyzer = DatasetAnalyzer()
    
    # Ejecutar análisis completo
    video_details = analyzer.analyze_videos()
    df = analyzer.create_balance_report(video_details)
    analyzer.create_visualizations(df)
    issues = analyzer.suggest_improvements(df)
    
    # Guardar reporte detallado
    df.to_csv('dataset_analysis_detailed.csv', index=False)
    print(f"\n💾 Reporte detallado guardado en 'dataset_analysis_detailed.csv'")
    
    return df, issues

if __name__ == "__main__":
    df, issues = run_analysis()
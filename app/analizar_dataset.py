import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# Configurar estilo para las gráficas
plt.style.use('default')
sns.set_palette("husl")

def analizar_dataset():
    """Analiza el balance del dataset de reconocimiento de gestos"""
    
    print("🔍 Analizando dataset_holistic.csv...")
    
    # Cargar el dataset
    try:
        df = pd.read_csv("data/dataset_holistic.csv")
        print(f"✅ Dataset cargado exitosamente")
        print(f"📊 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {e}")
        return
    
    # Obtener la columna de etiquetas (última columna)
    etiquetas = df.iloc[:, -1]
    
    print(f"\n🎯 Análisis de etiquetas:")
    print(f"📝 Total de muestras: {len(etiquetas)}")
    print(f"🏷️  Número de clases únicas: {etiquetas.nunique()}")
    
    # Contar frecuencias
    conteo = etiquetas.value_counts()
    print(f"\n📈 Distribución de clases:")
    print("=" * 50)
    
    for clase, cantidad in conteo.items():
        porcentaje = (cantidad / len(etiquetas)) * 100
        print(f"{clase:15} | {cantidad:4d} muestras | {porcentaje:5.1f}%")
    
    # Estadísticas de balance
    print(f"\n⚖️  Estadísticas de balance:")
    print("=" * 50)
    print(f"Mínimo muestras por clase: {conteo.min()}")
    print(f"Máximo muestras por clase: {conteo.max()}")
    print(f"Promedio muestras por clase: {conteo.mean():.1f}")
    print(f"Desviación estándar: {conteo.std():.1f}")
    
    # Calcular métricas de balance
    balance_ratio = conteo.min() / conteo.max()
    print(f"Ratio de balance (min/max): {balance_ratio:.3f}")
    
    # Evaluar el balance
    if balance_ratio >= 0.8:
        balance_status = "🟢 EXCELENTE"
    elif balance_ratio >= 0.6:
        balance_status = "🟡 BUENO"
    elif balance_ratio >= 0.4:
        balance_status = "🟠 REGULAR"
    else:
        balance_status = "🔴 DESBALANCEADO"
    
    print(f"Estado del balance: {balance_status}")
    
    # Crear visualizaciones
    crear_visualizaciones(conteo, etiquetas)
    
    # Recomendaciones
    generar_recomendaciones(conteo, balance_ratio)
    
    return conteo, balance_ratio

def crear_visualizaciones(conteo, etiquetas):
    """Crea gráficas para visualizar la distribución del dataset"""
    
    # Configurar el tamaño de las gráficas
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Análisis de Balance del Dataset', fontsize=16, fontweight='bold')
    
    # 1. Gráfica de barras de frecuencias
    axes[0, 0].bar(range(len(conteo)), conteo.values, color='skyblue', edgecolor='navy')
    axes[0, 0].set_title('📈 Distribución de Muestras por Clase')
    axes[0, 0].set_xlabel('Clases')
    axes[0, 0].set_ylabel('Número de Muestras')
    axes[0, 0].set_xticks(range(len(conteo)))
    axes[0, 0].set_xticklabels(conteo.index, rotation=45, ha='right')
    
    # Agregar valores en las barras
    for i, v in enumerate(conteo.values):
        axes[0, 0].text(i, v + max(conteo.values) * 0.01, str(v), 
                       ha='center', va='bottom', fontweight='bold')
    
    # 2. Gráfica de pastel
    axes[0, 1].pie(conteo.values, labels=conteo.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('🥧 Proporción de Clases')
    
    # 3. Gráfica de líneas para ver tendencias
    axes[1, 0].plot(range(len(conteo)), conteo.values, 'o-', linewidth=2, markersize=8)
    axes[1, 0].set_title('📉 Tendencia de Distribución')
    axes[1, 0].set_xlabel('Clases (ordenadas por frecuencia)')
    axes[1, 0].set_ylabel('Número de Muestras')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Histograma de frecuencias
    axes[1, 1].hist(conteo.values, bins=10, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    axes[1, 1].set_title('📊 Distribución de Frecuencias')
    axes[1, 1].set_xlabel('Número de Muestras por Clase')
    axes[1, 1].set_ylabel('Número de Clases')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Agregar línea de promedio
    promedio = conteo.mean()
    axes[1, 1].axvline(promedio, color='red', linestyle='--', linewidth=2, 
                       label=f'Promedio: {promedio:.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig('app/analisis_dataset.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráfica guardada como 'app/analisis_dataset.png'")
    
    # Mostrar la gráfica
    plt.show()

def generar_recomendaciones(conteo, balance_ratio):
    """Genera recomendaciones basadas en el análisis del dataset"""
    
    print(f"\n💡 RECOMENDACIONES:")
    print("=" * 50)
    
    # Clases con pocas muestras
    clases_pocas = conteo[conteo < conteo.mean() * 0.5]
    if len(clases_pocas) > 0:
        print(f"⚠️  Clases con pocas muestras (< 50% del promedio):")
        for clase, cantidad in clases_pocas.items():
            print(f"   - {clase}: {cantidad} muestras")
    
    # Clases con muchas muestras
    clases_muchas = conteo[conteo > conteo.mean() * 1.5]
    if len(clases_muchas) > 0:
        print(f"📈 Clases con muchas muestras (> 150% del promedio):")
        for clase, cantidad in clases_muchas.items():
            print(f"   - {clase}: {cantidad} muestras")
    
    # Recomendaciones específicas
    print(f"\n🎯 Recomendaciones para el entrenamiento:")
    
    if balance_ratio >= 0.8:
        print("✅ El dataset está bien balanceado. Puedes proceder con el entrenamiento normal.")
    elif balance_ratio >= 0.6:
        print("⚠️  El dataset tiene un balance moderado. Considera:")
        print("   - Usar técnicas de data augmentation")
        print("   - Ajustar los pesos de las clases en el modelo")
    else:
        print("🚨 El dataset está desbalanceado. Recomendaciones:")
        print("   - Recolectar más datos para las clases minoritarias")
        print("   - Usar técnicas de oversampling/undersampling")
        print("   - Implementar weighted loss function")
        print("   - Considerar técnicas de transfer learning")
    
    # Recomendaciones de hiperparámetros
    print(f"\n⚙️  Ajustes de hiperparámetros sugeridos:")
    print(f"   - Learning rate: {'0.001 (normal)' if balance_ratio >= 0.6 else '0.0005 (más bajo)'}")
    print(f"   - Batch size: {'32-64' if balance_ratio >= 0.6 else '16-32'}")
    print(f"   - Epochs: {'40-50' if balance_ratio >= 0.6 else '60-80'}")
    
    # Técnicas de evaluación
    print(f"\n📋 Métricas de evaluación recomendadas:")
    print("   - Accuracy general")
    print("   - Precision, Recall, F1-score por clase")
    print("   - Matriz de confusión")
    if balance_ratio < 0.6:
        print("   - ROC-AUC (especialmente importante para datasets desbalanceados)")

if __name__ == "__main__":
    print("🧠 ANALIZADOR DE BALANCE DE DATASET")
    print("=" * 50)
    
    # Verificar que el archivo existe
    if not os.path.exists("data/dataset_holistic.csv"):
        print("❌ No se encontró el archivo 'data/dataset_holistic.csv'")
        print("   Asegúrate de que el archivo esté en la carpeta correcta.")
    else:
        conteo, balance_ratio = analizar_dataset()
        
        print(f"\n🎉 Análisis completado!")
        print(f"📁 Resultados guardados en 'app/analisis_dataset.png'") 
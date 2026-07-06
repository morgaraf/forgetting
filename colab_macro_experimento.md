```python
# ==============================================================================
# PROYECTO DELFÍN - MACRO EXPERIMENTO FINAL: OLVIDO EXPLÍCITO Y HOMEOSTASIS
# ==============================================================================

# 1. Montamos Google Drive
from google.colab import drive
drive.mount('/content/drive')

import sys
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from scipy.stats import norm

# 2. Inyectamos la ruta del código oficial
ruta = '/content/drive/Othercomputers/Mi PC (1)/Proyecto DELFIN/fashion'
sys.path.insert(0, ruta)
importlib.invalidate_caches()

from associative import AssociativeMemory
import neural_net

print("\n--- INICIANDO MACRO-EXPERIMENTO ---")

# ==============================================================================
# FASE 1: PREPARACIÓN DE DATOS (70k -> 49k / 14k / 7k)
# ==============================================================================
print("1. Cargando y particionando Fashion MNIST...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Combinamos todo para hacer nuestras particiones exactas (70,000 en total)
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

x_all = x_all.astype('float32') / 255.0
x_all = np.expand_dims(x_all, -1)

# Mezclamos aleatoriamente antes de partir
np.random.seed(42)
indices = np.random.permutation(len(x_all))
x_all, y_all = x_all[indices], y_all[indices]

# Particiones según instrucciones del Profesor Rafael:
# 70% Entrenamiento Autoencoder (49,000)
x_auto = x_all[:49000]
# 20% Llenado de Memoria EAM (14,000)
x_memory = x_all[49000:63000]
y_memory = y_all[49000:63000]
# 10% Pruebas de Recall (7,000)
x_recall = x_all[63000:]
y_recall = y_all[63000:]

print(f"   -> Autoencoder: {len(x_auto)} imágenes")
print(f"   -> Memoria EAM: {len(x_memory)} imágenes")
print(f"   -> Recall Test: {len(x_recall)} imágenes")

# ==============================================================================
# FASE 2: ENTRENAMIENTO DEL AUTOENCODER VGG
# ==============================================================================
print("\n2. Construyendo y Entrenando el Autoencoder VGG...")
input_layer, encoder_output = neural_net.get_encoder()
encoder = Model(inputs=input_layer, outputs=encoder_output, name='VGG_Encoder')

decoder_input, decoder_output = neural_net.get_decoder()
decoder = Model(inputs=decoder_input, outputs=decoder_output, name='VGG_Decoder')

# Conectamos Encoder con Decoder
autoencoder = Model(inputs=input_layer, outputs=decoder(encoder_output))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamos con las 49,000 imágenes (Epochs bajos por tiempo en Colab, ajustable)
print("   (Entrenando pesos desde cero. Esto puede tomar unos minutos...)")
autoencoder.fit(x_auto, x_auto, epochs=5, batch_size=256, validation_split=0.1, verbose=1)

print("\n3. Extrayendo características (vectores latentes) para la Memoria...")
features_memory_raw = encoder.predict(x_memory, verbose=0)
features_recall_raw = encoder.predict(x_recall, verbose=0)

M_NIVELES = 16
N_COLUMNAS = features_memory_raw.shape[1]
f_min = np.min(features_memory_raw)
f_max = np.max(features_memory_raw)

# Discretización (0 a 15)
features_memory = np.round((features_memory_raw - f_min) / (f_max - f_min) * (M_NIVELES - 1)).astype(int)
features_recall = np.round((features_recall_raw - f_min) / (f_max - f_min) * (M_NIVELES - 1)).astype(int)

# ==============================================================================
# FASE 3: ALGORITMO DE DISTRIBUCIÓN PROBABILÍSTICA (Front-loaded a Back-loaded)
# ==============================================================================
print("\n4. Generando distribución probabilística de clases (Anclas)...")
# Separamos las imágenes por clase para poder muestrearlas
mem_by_class = {c: [] for c in range(10)}
for feat, label in zip(features_memory, y_memory):
    mem_by_class[label].append(feat)

ordered_memory_features = []
ordered_memory_labels = []

N_STEPS = 14000
# Clase 0 pica en t=0 (Front-loaded), Clase 9 pica en t=14000 (Back-loaded)
mu_classes = [c * (N_STEPS / 9.0) for c in range(10)]
sigma = 2500 # Dispersión de la campana

for t in range(N_STEPS):
    # Calculamos la probabilidad de cada clase en el instante t
    probs = np.array([norm.pdf(t, loc=mu, scale=sigma) for mu in mu_classes])
    
    # Ponemos en 0 la probabilidad de las clases que ya se quedaron sin imágenes
    for c in range(10):
        if len(mem_by_class[c]) == 0:
            probs[c] = 0.0
            
    probs = probs / np.sum(probs) # Normalizamos
    
    # Elegimos una clase basada en la distribución de probabilidad
    chosen_class = np.random.choice(10, p=probs)
    
    # Extraemos una imagen de esa clase
    img = mem_by_class[chosen_class].pop(0)
    ordered_memory_features.append(img)
    ordered_memory_labels.append(chosen_class)

# Gráfica de Verificación de Distribución
plt.figure(figsize=(10, 3))
plt.scatter(range(N_STEPS), ordered_memory_labels, alpha=0.1, s=1, c=ordered_memory_labels, cmap='tab10')
plt.title("Distribución Estocástica de Clases (0: Front-loaded -> 9: Back-loaded)")
plt.xlabel("Tiempo de Inserción (Paso)")
plt.ylabel("Clase (0-9)")
plt.show()

# ==============================================================================
# FASE 4: EXPERIMENTACIÓN CON OLVIDO EXPLÍCITO (W = 3500, 7000, 10500)
# ==============================================================================
valores_W = [3500, 7000, 10500]
colores_W = ['#e63946', '#2a9d8f', '#f4a261']

plt.figure(figsize=(12, 6))

for idx, W in enumerate(valores_W):
    print(f"\n--- Ejecutando Prueba con Límite de Energía W = {W} ---")
    eam = AssociativeMemory(n=N_COLUMNAS, m=M_NIVELES, max_col_weight=W)
    
    energia_col_10 = []
    
    for i, vector in enumerate(ordered_memory_features):
        eam.register(vector)
        energia_col_10.append(np.sum(eam._relation[:, 10]))
        
        if i % 3500 == 0 and i > 0:
            print(f"   Progreso: {i}/14000 imágenes procesadas...")

    print(f"-> Entropía Final (W={W}): {eam.entropy:.4f}")
    
    # Graficamos la curva para este W
    plt.plot(range(N_STEPS), energia_col_10, label=f"Dinámica con W = {W}", color=colores_W[idx], linewidth=2)
    plt.axhline(y=W, color=colores_W[idx], linestyle='--', alpha=0.5)

plt.title("Homeostasis Entrópica Comparativa: Límite de Capacidad W", fontsize=15, pad=15)
plt.xlabel("Cantidad de Imágenes Memorizadas (N_corpus = 14,000)", fontsize=12)
plt.ylabel("Energía Total de la Columna 10", fontsize=12)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n¡EXPERIMENTO FINALIZADO CON ÉXITO!")
```

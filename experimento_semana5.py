# ==============================================================================
# PROYECTO DELFÍN - SEMANA 5: EXPERIMENTO CORREGIDO
# Correcciones del Dr. Rafael Morales:
#   1. Distribución por BLOQUES LIMPIOS (sin traslape de clases 0 y 1)
#   2. Entrenamiento con 300 epochs + EarlyStopping (patience=15)
#   3. Guardar/cargar modelo y features en disco
# Uso: python experimento_semana5.py
# ==============================================================================

import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import matplotlib
import gzip

matplotlib.use('Agg')  # Backend sin GUI (funciona en servidores sin pantalla)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping

# Configurar ruta del código del proyecto
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_DIR = os.path.join(PROJECT_DIR, 'fashion')
sys.path.insert(0, FASHION_DIR)

from associative import AssociativeMemory
import neural_net

# Si el conjunto de datos se obtiene de Tensorflow, entonces DATA_DIR debe ser None
# DATA_DIR = None
DATA_DIR = os.path.join(PROJECT_DIR, 'data/fashion')

CLASS_NAMES = [
    'Camiseta',
    'Pantalón',
    'Suéter',
    'Vestido',
    'Abrigo',
    'Sandalia',
    'Camisa',
    'Zapatilla',
    'Bolsa',
    'Bota',
]

# Directorios para guardar modelos, features y resultados
SAVE_DIR = os.path.join(PROJECT_DIR, 'saved_models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'resultados')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ENCODER_PATH = os.path.join(SAVE_DIR, 'encoder_vgg.keras')
CLASSIFIER_PATH = os.path.join(SAVE_DIR, 'classifier.keras')
DECODER_PATH = os.path.join(SAVE_DIR, 'decoder_vgg.keras')
FEATURES_MEM_PATH = os.path.join(SAVE_DIR, 'features_memory.npy')
FEATURES_REC_PATH = os.path.join(SAVE_DIR, 'features_recall.npy')
LABELS_MEM_PATH = os.path.join(SAVE_DIR, 'labels_memory.npy')
LABELS_REC_PATH = os.path.join(SAVE_DIR, 'labels_recall.npy')

print('\n' + '=' * 70)
print('  PROYECTO DELFÍN - SEMANA 5: RECALL vs LLENADO POR CLASE')
print('  300 Epochs + EarlyStopping + Bloques Limpios')
print('  M = [4, 8, 16]  |  W = [∞, 10500, 7000, 3500]')
print('=' * 70)

# Detectar GPU
gpus = tf.config.list_physical_devices('GPU')
print(
    f'\n  Dispositivo: {"GPU (" + str(len(gpus)) + " encontradas)" if gpus else "CPU (sin GPU)"}'
)


# ==============================================================================
# FASE 1: PREPARACIÓN DE DATOS
# ==============================================================================
def load_data(path, kind):
    """Carga un conjunto de datos similar a MNIST desde `path`

    Parámetros
    ----------
    path : str
        Dirección del directorio de datos.
    kind : str
        'train' o 't10k' para datos de entrenamiento o de prueba."""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 28, 28
        )
    return images, labels


print('\n1. Cargando y particionando Fashion MNIST...')
if DATA_DIR is None:
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
else:
    # Cargar los datos desde DATA_DIR
    x_train, y_train = load_data(DATA_DIR, 'train')
    x_test, y_test = load_data(DATA_DIR, 't10k')

x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)
x_all = x_all.astype('float32') / 255.0
x_all = np.expand_dims(x_all, -1)

np.random.seed(42)
indices = np.random.permutation(len(x_all))
x_all, y_all = x_all[indices], y_all[indices]

# Particiones: 70% / 20% / 10%
x_auto = x_all[:49000]
y_auto = y_all[:49000]
x_memory = x_all[49000:63000]
y_memory = y_all[49000:63000]
x_recall = x_all[63000:]
y_recall = y_all[63000:]

print(f'   Autoencoder+Clasificador: {len(x_auto)}')
print(f'   Memoria EAM:              {len(x_memory)}')
print(f'   Pruebas Recall:           {len(x_recall)}')

# ==============================================================================
# FASE 2: ENTRENAMIENTO O CARGA DEL MODELO
# ==============================================================================
if os.path.exists(ENCODER_PATH) and os.path.exists(CLASSIFIER_PATH):
    print('\n2. ¡Modelo encontrado en disco! Cargando sin re-entrenar...')
    encoder = tf.keras.models.load_model(ENCODER_PATH)
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
    decoder = tf.keras.models.load_model(DECODER_PATH)
    print('   -> Encoder, Decoder y Clasificador cargados exitosamente.')
else:
    print('\n2. Modelo NO encontrado. Entrenando desde cero (300 epochs máx.)...')

    input_layer, encoder_output = neural_net.get_encoder()
    encoder = Model(inputs=input_layer, outputs=encoder_output, name='VGG_Encoder')

    decoder_input, decoder_output = neural_net.get_decoder()
    decoder = Model(inputs=decoder_input, outputs=decoder_output, name='VGG_Decoder')

    classifier_input, classifier_output = neural_net.get_classifier()
    classifier = Model(
        inputs=classifier_input, outputs=classifier_output, name='Classifier'
    )

    encoded = encoder(input_layer)
    decoded = decoder(encoded)
    classified = classifier(encoded)

    rmse_metric = tf.keras.metrics.RootMeanSquaredError()
    model = Model(inputs=input_layer, outputs=[classified, decoded])
    model.compile(
        loss=['categorical_crossentropy', 'mean_squared_error'],
        optimizer='adam',
        metrics={'Classifier': 'accuracy', 'VGG_Decoder': rmse_metric},
    )

    y_auto_cat = to_categorical(y_auto, num_classes=10)

    # EarlyStopping con patience=15 como recomienda el Dr. Rafael
    early_stop = EarlyStopping(
        monitor='val_Classifier_loss',
        patience=15,
        mode='min',
        restore_best_weights=True,
        verbose=2,
    )

    print('   Objetivo: val_Classifier_accuracy > 0.98, val_RMSE < 0.15')
    print('   (Con GPU ~1 hora, sin GPU ~4-6 horas)')

    history = model.fit(
        x_auto,
        (y_auto_cat, x_auto),
        epochs=300,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )

    # Reportar métricas finales
    final_val_acc = history.history['val_Classifier_accuracy'][-1]
    final_val_rmse = history.history['val_VGG_Decoder_root_mean_squared_error'][-1]
    total_epochs = len(history.history['loss'])
    print(f'\n   Entrenamiento finalizado en {total_epochs} epochs.')
    print(f'   val_Classifier_accuracy: {final_val_acc:.4f} (objetivo > 0.98)')
    print(f'   val_Decoder_RMSE:        {final_val_rmse:.4f} (objetivo < 0.15)')

    # Guardar modelos en disco
    print('\n   Guardando modelos entrenados en disco...')
    encoder.save(ENCODER_PATH)
    classifier.save(CLASSIFIER_PATH)
    decoder.save(DECODER_PATH)
    print(f'   -> Modelos guardados en: {SAVE_DIR}')

    # Guardar gráfica del entrenamiento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['Classifier_accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_Classifier_accuracy'], label='Val Accuracy')
    ax1.axhline(y=0.98, color='r', linestyle='--', alpha=0.5, label='Objetivo (0.98)')
    ax1.set_title('Accuracy del Clasificador')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['VGG_Decoder_root_mean_squared_error'], label='Train RMSE')
    ax2.plot(
        history.history['val_VGG_Decoder_root_mean_squared_error'], label='Val RMSE'
    )
    ax2.axhline(y=0.15, color='r', linestyle='--', alpha=0.5, label='Objetivo (0.15)')
    ax2.set_title('RMSE del Decoder')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'entrenamiento_curvas.png'), dpi=150)
    print(f'   -> Gráfica guardada en: {RESULTS_DIR}/entrenamiento_curvas.png')
    plt.close()

# ==============================================================================
# FASE 3: EXTRACCIÓN O CARGA DE CARACTERÍSTICAS
# ==============================================================================
if os.path.exists(FEATURES_MEM_PATH) and os.path.exists(FEATURES_REC_PATH):
    print('\n3. ¡Features encontradas en disco! Cargando...')
    features_memory_raw = np.load(FEATURES_MEM_PATH)
    features_recall_raw = np.load(FEATURES_REC_PATH)
    y_memory = np.load(LABELS_MEM_PATH)
    y_recall = np.load(LABELS_REC_PATH)
else:
    print('\n3. Extrayendo vectores latentes...')
    features_memory_raw = encoder.predict(x_memory, verbose=0)
    features_recall_raw = encoder.predict(x_recall, verbose=0)

    np.save(FEATURES_MEM_PATH, features_memory_raw)
    np.save(FEATURES_REC_PATH, features_recall_raw)
    np.save(LABELS_MEM_PATH, y_memory)
    np.save(LABELS_REC_PATH, y_recall)
    print(f'   -> Features guardadas en: {SAVE_DIR}')

N_COLUMNAS = features_memory_raw.shape[1]
print(f'   Dimensión latente: {N_COLUMNAS}')

# ==============================================================================
# FASE 4: DISTRIBUCIÓN POR BLOQUES LIMPIOS (CORREGIDA)
# Clase 0 SOLO al inicio, Clase 9 SOLO al final. SIN TRASLAPE.
# ==============================================================================
print('\n4. Generando distribución por BLOQUES LIMPIOS...')

idx_by_class = {c: [] for c in range(10)}
for i in range(len(y_memory)):
    idx_by_class[y_memory[i]].append(i)

np.random.seed(42)
ordered_indices = []
for c in range(10):
    class_indices = list(idx_by_class[c])
    np.random.shuffle(class_indices)
    ordered_indices.extend(class_indices)

y_memory = y_memory[ordered_indices]
features_memory_raw = features_memory_raw[ordered_indices]
N_STEPS = len(y_memory)

# Guardar gráfica de distribución temporal
plt.figure(figsize=(12, 3))
plt.scatter(range(N_STEPS), y_memory, alpha=0.3, s=1, c=y_memory, cmap='tab10')
plt.title('Distribución por Bloques Limpios (0: Front → 9: Back)', fontsize=13)
plt.xlabel('Paso de Inserción')
plt.ylabel('Clase')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'distribucion_temporal.png'), dpi=150)
print(f'   -> Gráfica guardada en: {RESULTS_DIR}/distribucion_temporal.png')
plt.close()


# ==============================================================================
# FASE 5: FUNCIÓN DE EVALUACIÓN DE RECALL
# ==============================================================================
def evaluar_recall_por_clase(
    eam, features_test_q, features_test_raw, y_test, classifier_model, M, f_min, f_max
):
    """Evalúa recall por clase. Retorna array de 10 tasas (una por clase)."""
    total = np.zeros(10)
    correctos = np.zeros(10)
    recalled_vectors = []
    recalled_true_labels = []

    for i in range(len(features_test_q)):
        true_label = y_test[i]
        total[true_label] += 1
        recalled_vec, accepted, weight = eam.recall(features_test_q[i])

        if accepted:
            recalled_clean = np.copy(recalled_vec).astype(float)
            nan_mask = np.isnan(recalled_clean)
            recalled_clean[nan_mask] = features_test_raw[i][nan_mask]
            recalled_float = (recalled_clean / (M - 1)) * (f_max - f_min) + f_min
            recalled_vectors.append(recalled_float)
            recalled_true_labels.append(true_label)

    if len(recalled_vectors) > 0:
        batch = np.array(recalled_vectors)
        predictions = classifier_model.predict(batch, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        for pred, true_l in zip(predicted_labels, recalled_true_labels):
            if pred == true_l:
                correctos[true_l] += 1

    recall_por_clase = np.where(total > 0, correctos / total * 100, 0)
    return recall_por_clase


# ==============================================================================
# FASE 6: MEGA-EXPERIMENTO (3 valores de M × 4 valores de W)
# ==============================================================================
valores_M = [4, 8, 16]
valores_W = [None, 10500, 7000, 3500]
etiquetas_W = ['W = ∞ (sin olvido)', 'W = 10,500', 'W = 7,000', 'W = 3,500']
colores_W = ['#264653', '#2a9d8f', '#e9c46a', '#e63946']

checkpoints = list(range(1000, 14001, 1000))

resultados = {}

for M in valores_M:
    print(f'\n{"#" * 70}')
    print(f'  EXPERIMENTANDO CON M = {M} NIVELES')
    print(f'{"#" * 70}')

    f_min = np.min(features_memory_raw)
    f_max = np.max(features_memory_raw)

    features_mem_q = np.round(
        (features_memory_raw - f_min) / (f_max - f_min) * (M - 1)
    ).astype(int)
    features_mem_q = np.clip(features_mem_q, 0, M - 1)

    features_rec_q = np.round(
        (features_recall_raw - f_min) / (f_max - f_min) * (M - 1)
    ).astype(int)
    features_rec_q = np.clip(features_rec_q, 0, M - 1)

    resultados[M] = {}

    for idx_w, W in enumerate(valores_W):
        W_label = etiquetas_W[idx_w]
        print(f'\n   --- M={M}, {W_label} ---')

        eam = AssociativeMemory(n=N_COLUMNAS, m=M, max_col_weight=W)
        recall_history = []

        for step in range(N_STEPS):
            eam.register(features_mem_q[step])

            if (step + 1) in checkpoints:
                print(f'      Checkpoint {step + 1}/14000 - Evaluando recall...')
                recall_clase = evaluar_recall_por_clase(
                    eam,
                    features_rec_q,
                    features_recall_raw,
                    y_recall,
                    classifier,
                    M,
                    f_min,
                    f_max,
                )
                recall_history.append(recall_clase)
                avg = np.mean(recall_clase)
                print(f'        Recall promedio: {avg:.1f}%')

        resultados[M][W] = np.array(recall_history)
        print(f'   -> Entropía final (M={M}, {W_label}): {eam.entropy:.4f}')

    # ==========================================================================
    # GRÁFICAS: 10 subplots (una por clase)
    # ==========================================================================
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    fig.suptitle(
        f'Proyecto Delfín: Recall por Clase durante Llenado (M = {M})',
        fontsize=16,
        fontweight='bold',
        y=1.02,
    )

    for c in range(10):
        ax = axes[c // 5][c % 5]
        for idx_w, W in enumerate(valores_W):
            curva = resultados[M][W][:, c]
            ax.plot(
                checkpoints,
                curva,
                marker='o',
                markersize=4,
                color=colores_W[idx_w],
                linewidth=2,
                label=etiquetas_W[idx_w],
            )
        ax.set_title(f'Clase {c}: {CLASS_NAMES[c]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Imágenes insertadas', fontsize=9)
        ax.set_ylabel('Recall (%)', fontsize=9)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, f'recall_por_clase_M{M}.png'),
        dpi=150,
        bbox_inches='tight',
    )
    print(f'   -> Gráfica guardada: {RESULTS_DIR}/recall_por_clase_M{M}.png')
    plt.close()

    # Gráfica resumen: Recall PROMEDIO
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for idx_w, W in enumerate(valores_W):
        curva_promedio = np.mean(resultados[M][W], axis=1)
        ax2.plot(
            checkpoints,
            curva_promedio,
            marker='s',
            markersize=5,
            color=colores_W[idx_w],
            linewidth=2.5,
            label=etiquetas_W[idx_w],
        )
    ax2.set_title(
        f'Recall Promedio Global durante Llenado (M = {M})',
        fontsize=14,
        fontweight='bold',
    )
    ax2.set_xlabel('Imágenes Insertadas', fontsize=12)
    ax2.set_ylabel('Recall Promedio (%)', fontsize=12)
    ax2.set_ylim(-5, 105)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'recall_promedio_M{M}.png'), dpi=150)
    print(f'   -> Gráfica guardada: {RESULTS_DIR}/recall_promedio_M{M}.png')
    plt.close()

# ==============================================================================
# TABLA RESUMEN FINAL
# ==============================================================================
print('\n' + '=' * 70)
print('  TABLA RESUMEN FINAL: RECALL (%) AL 100% DE LLENADO')
print('=' * 70)

# Guardar tabla en archivo de texto
tabla_path = os.path.join(RESULTS_DIR, 'tabla_resumen.txt')
with open(tabla_path, 'w', encoding='utf-8') as f:
    for M in valores_M:
        header_line = f'\n  --- M = {M} ---\n'
        print(header_line)
        f.write(header_line)

        header = f'  {"Clase":<12}'
        for W_label in etiquetas_W:
            header += f' {W_label:>18}'
        print(header)
        f.write(header + '\n')

        sep = f'  {"-" * 80}'
        print(sep)
        f.write(sep + '\n')

        for c in range(10):
            row = f'  {c} {CLASS_NAMES[c]:<10}'
            for W in valores_W:
                val = resultados[M][W][-1, c]
                row += f' {val:>17.1f}%'
            print(row)
            f.write(row + '\n')

        print(sep)
        f.write(sep + '\n')

        row_avg = f'  {"PROMEDIO":<12}'
        for W in valores_W:
            avg = np.mean(resultados[M][W][-1, :])
            row_avg += f' {avg:>17.1f}%'
        print(row_avg)
        f.write(row_avg + '\n')

print(f'\n   -> Tabla guardada en: {tabla_path}')

print('\n' + '=' * 70)
print('  ¡EXPERIMENTO SEMANA 5 FINALIZADO CON ÉXITO!')
print(f'  Todos los resultados guardados en: {RESULTS_DIR}')
print('=' * 70)

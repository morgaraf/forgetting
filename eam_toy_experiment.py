import tensorflow as tf
import matplotlib.pyplot as plt

def run_toy_experiment(eam_layer):
    """
    Simula un proceso de aprendizaje continuo en el que los pesos de la memoria crecen.
    El mecanismo EAM entra en acción para acotar la energía.
    Ideal para visualizar el comportamiento antes de pasar a datasets complejos como MNIST.
    """
    print("Iniciando experimento de simulación EAM...")
    
    # 1. Inicializamos una matriz de memoria vacía de 10x10
    memory_matrix = tf.zeros([10, 10], dtype=tf.float32)
    
    # Listas para guardar el historial y poder graficar
    historial_energia_col_5 = []
    
    pasos_tiempo = 100
    
    for step in range(pasos_tiempo):
        # Fase de Aprendizaje (Simulada)
        # Generamos ruido aleatorio para simular entradas de datos que fortalecen conexiones
        activaciones = tf.random.uniform([10, 10], minval=0.0, maxval=0.5, dtype=tf.float32)
        
        # Determinamos qué celdas se "activaron" en este paso (ej. valor > 0.2)
        active_cells_mask = activaciones > 0.2
        
        # Incrementamos el peso de las conexiones activas (Aprendizaje tipo Hebbiano)
        incremento = tf.cast(active_cells_mask, tf.float32) * 0.4
        memory_matrix = memory_matrix + incremento
        
        # Fase de Olvido (Homeostasis)
        # Aplicamos la capa que creamos para asegurar que no se exceda la energía máxima
        memory_matrix = eam_layer(memory_matrix, active_cells_mask)
        
        # Registramos la energía de una columna específica (la columna 5) para monitorear
        energia_columna_5 = tf.reduce_sum(memory_matrix[:, 5]).numpy()
        historial_energia_col_5.append(energia_columna_5)
        
    print("Simulación terminada. Generando gráfica...")
    
    # Graficamos los resultados
    plt.figure(figsize=(10, 5))
    plt.plot(range(pasos_tiempo), historial_energia_col_5, label="Energía de la Columna", color="#008080", linewidth=2.5)
    plt.axhline(y=eam_layer.max_energy_threshold.numpy(), color='#FF0000', linestyle='--', linewidth=2, label=f"Umbral de Energía Permitida ({eam_layer.max_energy_threshold.numpy()})")
    
    plt.title("Proyecto Delfín: Dinámica de Conservación de Energía EAM", fontsize=14, pad=15)
    plt.xlabel("Pasos de Aprendizaje (Iteraciones)", fontsize=12)
    plt.ylabel("Energía Total Acumulada", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Retornamos la figura por si se necesita manipular en Colab
    return plt.gcf()

if __name__ == "__main__":
    # Prueba rápida si se ejecuta localmente (no mostrará el plot bonito de Colab, pero no dará error)
    print("Ejecuta esto desde Colab para ver los gráficos.")

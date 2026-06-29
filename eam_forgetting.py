import tensorflow as tf

class EAMForgetting(tf.keras.layers.Layer):
    """
    Capa de Olvido para la Memoria Asociativa Entrópica (EAM).
    
    Implementa un mecanismo de olvido explícito basado en la conservación de energía.
    La "energía" se define como la suma de los pesos (conexiones) dentro de una columna
    específica de la matriz de memoria 2D.
    
    Si la energía total de una columna excede el `max_energy_threshold`, se aplica
    un decaimiento (tasa de olvido) a las celdas activas para mantener el equilibrio
    homeostático, garantizando que ningún peso adopte valores negativos.
    """
    
    def __init__(self, max_energy_threshold: float, forgetting_rate: float, **kwargs):
        """
        Inicializa el mecanismo de olvido.
        
        Args:
            max_energy_threshold (float): Límite máximo de energía permitida por columna.
            forgetting_rate (float): Cantidad de energía que se resta cuando se supera el límite.
            **kwargs: Argumentos adicionales para tf.keras.layers.Layer.
        """
        super(EAMForgetting, self).__init__(**kwargs)
        self.max_energy_threshold = tf.constant(max_energy_threshold, dtype=tf.float32)
        self.forgetting_rate = tf.constant(forgetting_rate, dtype=tf.float32)

    @tf.function
    def call(self, memory_matrix: tf.Tensor, active_cells_mask: tf.Tensor = None) -> tf.Tensor:
        """
        Aplica el mecanismo de olvido vectorizado a la matriz de memoria.
        
        Args:
            memory_matrix (tf.Tensor): Matriz de memoria 2D de forma (filas, columnas).
            active_cells_mask (tf.Tensor, opcional): Matriz booleana 2D de la misma forma
                que `memory_matrix`. Indica qué celdas estuvieron activas o fueron
                actualizadas. Si es None, el olvido se aplica a toda la columna que exceda el límite.
                
        Returns:
            tf.Tensor: Matriz de memoria actualizada, asegurando que no haya valores negativos.
        """
        # 1. Monitorear la energía total (suma de pesos) de cada columna.
        # Sumamos a lo largo del eje 0 (filas) para obtener la energía por columna.
        column_energies = tf.reduce_sum(memory_matrix, axis=0)
        
        # 2. Establecer un límite máximo de energía permitida por columna.
        # Determinamos qué columnas han superado el umbral.
        exceeds_threshold = column_energies > self.max_energy_threshold
        
        # Convertimos la máscara booleana a float32 para operaciones matemáticas
        # y la reformateamos para permitir el broadcasting (1 fila, N columnas).
        exceeds_threshold_float = tf.cast(exceeds_threshold, dtype=tf.float32)
        exceeds_threshold_float = tf.reshape(exceeds_threshold_float, [1, -1])
        
        # 3. Restar una tasa de olvido a las celdas (activas o todas).
        if active_cells_mask is not None:
            # Solo aplicamos el olvido a las celdas que fueron marcadas como activas
            active_cells_float = tf.cast(active_cells_mask, dtype=tf.float32)
            penalty = self.forgetting_rate * exceeds_threshold_float * active_cells_float
        else:
            # Si no hay máscara, el olvido afecta a toda la columna en infracción
            penalty = self.forgetting_rate * exceeds_threshold_float
            
        # Aplicamos la penalización a la memoria
        updated_memory = memory_matrix - penalty
        
        # 4. Garantizar que no existan valores de energía negativos en la matriz.
        # Utilizamos tf.maximum para aplicar ReLU(x) o max(x, 0) elemento a elemento.
        updated_memory = tf.maximum(updated_memory, 0.0)
        
        return updated_memory

# =====================================================================
# Script Puente / Ejemplo de Uso para Google Colab
# =====================================================================
def run_bridge_test():
    """
    Función de prueba para validar el comportamiento antes de integrarlo 
    al repositorio principal ("imagine").
    """
    # Configuramos parámetros iniciales
    UMBRAL_MAX_ENERGIA = 10.0
    TASA_OLVIDO = 2.0
    
    # Instanciamos el mecanismo
    eam_forgetting = EAMForgetting(
        max_energy_threshold=UMBRAL_MAX_ENERGIA, 
        forgetting_rate=TASA_OLVIDO
    )
    
    # Creamos una matriz de memoria ficticia (ej. 4 filas x 3 columnas)
    # Columna 0: Suma = 6.0 (No supera)
    # Columna 1: Suma = 11.0 (Supera el umbral de 10.0)
    # Columna 2: Suma = 15.0 (Supera el umbral de 10.0)
    memory_matrix = tf.constant([
        [1.0, 4.0, 5.0],
        [2.0, 3.0, 5.0],
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 2.0]
    ], dtype=tf.float32)
    
    # Simulamos cuáles celdas se activaron recientemente (True)
    active_cells_mask = tf.constant([
        [False, True,  True],
        [False, False, True],
        [True,  True,  False],
        [False, False, True]
    ], dtype=tf.bool)
    
    print("--- Estado Inicial ---")
    print("Matriz de Memoria:\n", memory_matrix.numpy())
    print("Energía por Columna:", tf.reduce_sum(memory_matrix, axis=0).numpy())
    print("\nUmbral de Energía Permitida:", UMBRAL_MAX_ENERGIA)
    print("Tasa de Olvido a aplicar:", TASA_OLVIDO)
    
    # Ejecutamos el mecanismo
    updated_memory = eam_forgetting(memory_matrix, active_cells_mask)
    
    print("\n--- Estado Final (Post-Olvido) ---")
    print("Matriz de Memoria Actualizada:\n", updated_memory.numpy())
    print("Nueva Energía por Columna:", tf.reduce_sum(updated_memory, axis=0).numpy())

if __name__ == "__main__":
    run_bridge_test()

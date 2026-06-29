import numpy as np

def lambda_register_update(memory_matrix: np.ndarray, 
                           input_vector: np.ndarray, 
                           max_energy: float, 
                           forgetting_rate: float,
                           top_k: int = 3) -> np.ndarray:
    """
    Actualiza la matriz de memoria (lambda-register) e incorpora el mecanismo
    de olvido explícito por conservación de energía para evitar saturación (alta entropía).

    Fundamentos Matemáticos:
    ------------------------
    Sea M la matriz de memoria de dimensión (N, M). En la etapa lambda-register,
    la actualización estándar es aditiva. Luego, se evalúa la energía por columna:
        E_j = \sum_{i=1}^{N} M_{i,j}
    Si E_j > E_{max}, se aplica una penalización \gamma a las top-k celdas más activas,
    redistribuyendo el efecto para mantener la homeostasis, con restricción M_{i,j} >= 0.

    Args:
        memory_matrix (np.ndarray): Matriz 2D de memoria EAM (N filas x M columnas).
        input_vector (np.ndarray): Vector o matriz de actualización a sumar (N x M).
        max_energy (float): Límite máximo de energía (suma de pesos) permitida por columna.
        forgetting_rate (float): Tasa de olvido (cuánto restar a las celdas activas).
        top_k (int): Número de celdas "más activas" a penalizar en caso de exceso.

    Returns:
        np.ndarray: Matriz de memoria actualizada y regulada energéticamente.
    """
    # 1. Operación lambda-register estándar: Suma matricial
    # (Se asume que input_vector ya ha sido mapeado/escalado por el autoencoder)
    updated_memory = memory_matrix + input_vector
    
    # 2. Monitorear la energía total (suma de pesos) de cada columna
    column_energies = np.sum(updated_memory, axis=0)
    
    # Identificar qué columnas han superado el límite de energía
    exceeding_columns_idx = np.where(column_energies > max_energy)[0]
    
    # 3. Intervención de Olvido (Homeostasis)
    for col_idx in exceeding_columns_idx:
        col_data = updated_memory[:, col_idx]
        
        # Identificar las celdas más activas (las de mayor peso)
        # Usamos argpartition para eficiencia (o argsort si k es pequeño)
        active_indices = np.argsort(col_data)[-top_k:]
        
        # Restar la tasa de olvido a estas celdas y "redistribuir" el efecto
        # En este contexto, redistribuir significa aplicar la penalización ponderada
        # o repartir la tasa de olvido equitativamente entre las celdas más activas.
        # Aquí aplicamos el forgetting_rate a las top_k celdas.
        penalty_per_cell = forgetting_rate / len(active_indices)
        col_data[active_indices] -= penalty_per_cell
        
        # Volver a asignar la columna (el array col_data es una vista, 
        # pero es buena práctica de legibilidad asegurarlo)
        updated_memory[:, col_idx] = col_data
        
    # 4. Garantizar que no existan valores de energía negativos (ReLU equivalente)
    # Cualquier celda que haya bajado de 0 se estabiliza en 0.
    updated_memory = np.maximum(updated_memory, 0.0)
    
    return updated_memory

# =====================================================================
# Script Puente / Ejemplo de Uso para Google Colab
# =====================================================================
def run_bridge_test():
    """
    Función para validar el comportamiento NumPy antes de integrar
    en el entorno principal de experimentación (Colab).
    """
    # Configuración
    np.random.seed(42)
    MAX_ENERGY = 10.0
    FORGETTING_RATE = 2.0
    
    # Simulamos una matriz 5x4 y un input entrante
    memory_matrix = np.array([
        [1.0, 4.0, 5.0, 0.5],
        [2.0, 3.0, 4.0, 1.0],
        [1.0, 2.0, 3.0, 0.0],
        [2.0, 2.0, 2.0, 0.5],
        [0.5, 0.5, 0.5, 0.5]
    ], dtype=np.float32)
    
    # Simulamos una entrada que causará que la Columna 2 (índice 2) supere el umbral
    input_vector = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    print("=== Estado Antes de lambda-register ===")
    print(f"Memoria Inicial:\n{memory_matrix}")
    print(f"Energía por Columna: {np.sum(memory_matrix, axis=0)}\n")
    
    # Ejecutamos la función
    updated_memory = lambda_register_update(
        memory_matrix=memory_matrix,
        input_vector=input_vector,
        max_energy=MAX_ENERGY,
        forgetting_rate=FORGETTING_RATE,
        top_k=2 # Afectar a las 2 celdas más activas
    )
    
    print(f"=== Estado Después de lambda-register ===")
    print(f"Memoria Actualizada:\n{updated_memory}")
    print(f"Nueva Energía por Columna: {np.sum(updated_memory, axis=0)}")
    print(f"Nota: Ningún valor es negativo.")

if __name__ == "__main__":
    run_bridge_test()

import numpy as np
import matplotlib.pyplot as plt

def lambda_register_update(memory_matrix: np.ndarray, 
                           input_vector: np.ndarray, 
                           max_energy: float, 
                           forgetting_rate: float,
                           top_k: int = 3) -> np.ndarray:
    """
    Actualiza la matriz de memoria e incorpora el mecanismo de olvido
    explícito por conservación de energía.
    """
    updated_memory = memory_matrix + input_vector
    column_energies = np.sum(updated_memory, axis=0)
    exceeding_columns_idx = np.where(column_energies > max_energy)[0]
    
    for col_idx in exceeding_columns_idx:
        col_data = updated_memory[:, col_idx]
        active_indices = np.argsort(col_data)[-top_k:]
        penalty_per_cell = forgetting_rate / len(active_indices)
        col_data[active_indices] -= penalty_per_cell
        updated_memory[:, col_idx] = col_data
        
    updated_memory = np.maximum(updated_memory, 0.0)
    return updated_memory

def run_numpy_toy_experiment():
    """
    Simula un proceso de aprendizaje continuo utilizando la implementación
    basada en NumPy y genera una visualización gráfica de la conservación de energía.
    """
    print("Iniciando experimento de simulación EAM con NumPy...")
    np.random.seed(42)
    
    # Parámetros del experimento
    pasos_tiempo = 100
    UMBRAL_MAX_ENERGIA = 10.0
    TASA_OLVIDO = 2.0
    TOP_K = 3
    
    # Matriz de memoria inicial vacía (10x10)
    memory_matrix = np.zeros((10, 10), dtype=np.float32)
    
    historial_energia_col_5 = []
    
    for step in range(pasos_tiempo):
        # 1. Simular entrada del Autoencoder (ruido aleatorio)
        activaciones = np.random.uniform(low=0.0, high=0.5, size=(10, 10))
        
        # Filtramos para simular esparcidad en las conexiones entrantes (como hace un ReLU)
        input_vector = np.where(activaciones > 0.2, activaciones * 0.8, 0.0)
        
        # 2. Aplicar el mecanismo lambda-register con olvido
        memory_matrix = lambda_register_update(
            memory_matrix=memory_matrix, 
            input_vector=input_vector, 
            max_energy=UMBRAL_MAX_ENERGIA, 
            forgetting_rate=TASA_OLVIDO,
            top_k=TOP_K
        )
        
        # 3. Registrar la energía de la columna 5 para monitoreo
        energia_columna_5 = np.sum(memory_matrix[:, 5])
        historial_energia_col_5.append(energia_columna_5)
        
    print("Simulación terminada. Generando gráfica...")
    
    # Graficar los resultados
    plt.figure(figsize=(10, 5))
    plt.plot(range(pasos_tiempo), historial_energia_col_5, label="Energía de la Columna 5", color="#2a9d8f", linewidth=2.5)
    plt.axhline(y=UMBRAL_MAX_ENERGIA, color='#e76f51', linestyle='--', linewidth=2, 
                label=f"Umbral de Energía Permitida ({UMBRAL_MAX_ENERGIA})")
    
    plt.title("Proyecto Delfín: Dinámica de Conservación de Energía EAM (NumPy)", fontsize=14, pad=15)
    plt.xlabel("Pasos de Aprendizaje (Iteraciones)", fontsize=12)
    plt.ylabel("Energía Total Acumulada", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Retornar y mostrar la figura
    plt.show()

if __name__ == "__main__":
    run_numpy_toy_experiment()

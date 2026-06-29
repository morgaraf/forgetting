import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Agregamos la ruta de fashion para importar el modulo oficial
sys.path.append(os.path.join(os.path.dirname(__file__), 'fashion'))

from associative import AssociativeMemory

def run_test():
    print("Probando el olvido explícito en el código oficial...")
    np.random.seed(42)
    
    # Parámetros
    pasos = 100
    UMBRAL_W = 15 # Límite W (max_col_weight)
    
    # Instanciamos la memoria oficial: 10 propiedades (n), 10 valores posibles (m)
    # Por defecto m en EAM se expande a m+1 para el valor indefinido.
    # Le pasamos nuestro nuevo parámetro max_col_weight.
    memory = AssociativeMemory(n=10, m=10, max_col_weight=UMBRAL_W)
    
    historial_energia_col_5 = []
    
    for _ in range(pasos):
        # Generamos un vector latente aleatorio (cada propiedad toma un valor entre 0 y 9)
        # Algunos valores pueden ser indefinidos, pero para forzar el llenado, 
        # enviamos valores completos.
        vector = np.random.randint(0, 10, size=10)
        
        # Simulamos que ciertas conexiones son más frecuentes
        if np.random.rand() > 0.5:
            vector[5] = 3 # Frecuentemente la columna 5 toma el valor 3
            
        # Registramos en la memoria
        memory.register(vector)
        
        # Extraemos la energía (suma de pesos) de la columna 5
        # Recordatorio: memory._relation tiene m+1 filas
        energia_col_5 = np.sum(memory._relation[:, 5])
        historial_energia_col_5.append(energia_col_5)
        
    print("Simulación terminada.")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(pasos), historial_energia_col_5, label="Energía de la Columna 5", color="#2a9d8f", linewidth=2.5)
    plt.axhline(y=UMBRAL_W, color='#e76f51', linestyle='--', linewidth=2, 
                label=f"Límite W = {UMBRAL_W}")
    
    plt.title("Prueba de Olvido Explícito EAM (Código Oficial)", fontsize=14, pad=15)
    plt.xlabel("Pasos de Aprendizaje (Iteraciones)")
    plt.ylabel("Energía Total (Suma de pesos en la columna)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Guardar la figura
    plt.savefig("test_olvido.png")
    print("Gráfica guardada en test_olvido.png")

if __name__ == "__main__":
    run_test()

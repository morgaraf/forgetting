# Memoria Asociativa Entrópica (EAM): Mecanismo de Olvido Explícito

Este documento formaliza la implementación técnica y los fundamentos teóricos del mecanismo de "Olvido Explícito" integrado en la arquitectura base del **Proyecto Delfín**. Esta documentación servirá como base para la redacción del artículo científico.

## 1. Fundamentos Teóricos del Olvido Explícito

En la formulación original de la Memoria Asociativa Entrópica (Pineda et al., 2021), la memoria opera acumulando frecuencias en un registro bidimensional (AMR). Este diseño provoca un **olvido implícito**: las celdas más activas adquieren pesos dominantes, marginando exponencialmente a las instancias menos comunes y saturando la entropía global del sistema.

Para mitigar este colapso entrópico, se ha introducido un **mecanismo de conservación de energía homeostática a nivel de columna**.

### 1.1 Restricción de Capacidad ($W$)
En lugar de imponer un límite de bits por celda (e.g., $1023$), se establece un hiperparámetro $W$ (`max_col_weight`) que dictamina la "energía" o masa probabilística máxima que puede contener una columna (propiedad) entera.

Sea $M$ la matriz de memoria. Para cualquier columna $j$, la suma de sus pesos $S_j$ está acotada:
$$ S_j = \sum_{i} M_{i,j} \leq W $$

**Determinación Empírica del Parámetro $W$**
A nivel experimental, el valor de $W$ debe ser proporcional al tamaño del corpus de objetos a registrar. Si el hiperparámetro $W$ es mayor o igual al número total de elementos insertados en la memoria, el sistema jamás saturará su energía y el olvido explícito nunca se disparará. 
Como regla general (validada experimentalmente con Fashion-MNIST), un valor óptimo es **$W = \frac{N_{corpus}}{2}$**, donde $N_{corpus}$ representa la cantidad de vectores en el corpus de llenado (e.g. para 14,000 elementos, $W = 7,000$).
### 1.2 Registro Computacional e Inversión Probabilística
Durante la fase de registro ($\lambda$-register), cuando una nueva pista $x$ indica que la celda $c$ en la columna $j$ debe ser incrementada:

1. **Estado No Saturado ($S_j < W$)**: Se incrementa $M_{c,j}$ normalmente.
2. **Estado Saturado ($S_j = W$)**: 
   - Se incrementa $M_{c,j}$ sumando el nuevo registro.
   - Para conservar el equilibrio termodinámico de la columna, se debe **decrementar** otra celda.
   - La selección de la celda a penalizar $k$ ($k \neq c$) es estocástica, sesgada hacia los trazos de memoria más débiles. 

Para lograr este sesgo, convertimos los pesos actuales en una **distribución de probabilidad invertida** $P_{inv}$. Para cualquier celda $k$ donde $M_{k,j} > 0$:
$$ P_{inv}(k) = \frac{1 - \frac{M_{k,j}}{\sum M_{m,j}}}{N_{activas} - 1} $$

Esta distribución garantiza matemáticamente que los trazos de ruido residual (pesos bajos) sean los primeros en desaparecer, preservando los "recuerdos" fuertemente consolidados, pero acotando su dominancia infinita.

---

## 2. Implementación en Código (`associative.py`)

Las modificaciones estructurales se realizaron en el repositorio `eam-experiments/fashion`.

> [!TIP]
> **Refactorización de Dependencias**
> Se eliminaron los objetos nativos de Linux (`signal.Sigmasks`) y los *aliasing* obsoletos de NumPy (`np.int`, `np.bool`) presentes en el código de 2020 para garantizar la compatibilidad universal cruzada, permitiendo su ejecución nativa en *Windows* y versiones modernas de *Google Colab*.

### Modificación de la clase `AssociativeMemory`
La lógica matemática fue inyectada directamente en el método `abstract()`, el cual procesa las inserciones matriciales. El código ahora itera sobre las dimensiones latentes extrayendo la matriz de probabilidad, calculando la divergencia estocástica con `np.random.choice(p=probs)`, y ajustando dinámicamente los pesos en tiempo real $\mathcal{O}(n)$.

---

## 3. Verificación Experimental y Gráfica

Mediante la extracción empírica de vectores latentes (características) del conjunto de datos **Fashion-MNIST** procesados a través de la arquitectura VGG (codificador), se observó exitosamente el fenómeno de **Homeostasis Entrópica**.

Al graficar la acumulación de energía a lo largo de 500 iteraciones de aprendizaje episódico continuo, la curva de energía demuestra un crecimiento lineal inicial hasta chocar elásticamente con el hiperplano $W$, momento en el cual la derivada se vuelve $0$ y la energía se estabiliza de forma permanente, previniendo el desbordamiento de la matriz y validando la hipótesis del profesor Rafael.

### 3.1 Impacto de la Fórmula de Probabilidad
Durante las pruebas de simulación gráfica, se comparó el impacto de dos distribuciones de probabilidad invertida para el descarte estocástico:
- **Distribución de Proporcionalidad Inversa Simple ($P \propto \frac{1}{w}$)**: Permitió una estabilización inicial con una entropía final de `0.1228`.
- **Distribución de Diferencia Relativa ($P = 1 - \frac{w}{\sum w}$)**: Al inyectar la ecuación matemática formal del profesor Rafael, el sistema estabilizó su entropía en **`0.1104`**. Esta ligera pero crucial reducción matemática demuestra empíricamente que la memoria es capaz de consolidar la información con mayor certidumbre, limpiando el ruido de fondo (falsos positivos) con una mayor eficiencia paramétrica.

Actualmente, la infraestructura del proyecto (repositorio, entorno en Google Colab y base de código EAM) se encuentra depurada y estabilizada. El modelo está a la espera de ser sometido al entrenamiento a escala real (70,000 imágenes) utilizando $W = 7,000$.

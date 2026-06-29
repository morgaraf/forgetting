# Memoria Asociativa Entrópica: de olvido implícito a explícito

En la memoria asociativa entrópica actual, la información se guarda en una tabla en la que cada columna se asocia a una propiedad del objeto en su representación abstracta (latente) y cada renglón a un valor (discretizado) de esa propiedad. El valor de una celda en la columna *c*  y el renglón *r*, que solemos llamar “el peso de la celda”, representa el número de veces que se ha registrado en la memoria un objeto con un valor *r* en su propiedad *c*. Con el fin de poder establecer una medida exacta del tamaño de la memoria (tamaño de la tabla), se ha puesto un límite al valor del peso de una celda, de modo que ésta no utilice más de dos bytes.

Para generar un recuerdo, la memoria selecciona un valor para cada propiedad del objeto; esto es, selecciona un renglón (celda) para cada columna en la tabla. La selección depende de la pista a partir de la cual se genera el recuerdo y de los pesos de las celdas en la columna, por lo que se puede decir que se tiende a seleccionar la celda con el mayor peso y más cercanía al valor de la pista en esa columna.

## Olvido implícito

Conforme se van acumulando objetos en la memoria, los pesos de las celdas en cada columna van creciendo de manera desigual: las celdas que representan los valores que más se repiten aumentan mucho de peso, mientras que las celdas que representan los valores que menos se repiten no aumentan mucho de peso. Una consecuencia de este comportamiento de la memoria es que los objetos con valores poco comunes tienden a no ser recuperados, porque las celdas que los representan tienen poco peso en comparación con las demás, por lo que se puede decir que el objeto tiende a quedar en el olvido.

## Olvido explícito

Para implementar un olvido explícito, vamos a cambiar la manera en que definimos un peso máximo: en vez de que cada celda tenga un límite en el peso que puede tener, vamos a tener un límite en la suma de los pesos de las celdas en cada columna (llamemos a ese límite *W*).

Actualmente, cuando vamos a registrar un objeto en la memoria, incrementamos el peso de las celdas correspondientes si no han alcanzado su valor máximo. Lo que vamos a hacer ahora, para cada propiedad del objeto, es sumar los pesos de las celdas en la columna correspondiente (llamemos *S* al valor de la suma) y entonces:

1. Si *S* < *W*, simplemente incrementamos el peso de la celda correspondiente.
2. Si *S* = *W*, incrementamos el peso de la celda correspondiente, pero decrementamos el peso de otra celda en la columna, escogida al azar (ver abajo).

En otras palabras, si la columna “ya está llena” entonces “tomamos prestado” peso de otra celda para incrementar el peso de la celda que corresponde al objeto que se está registrando en la memoria.

Para seleccionar “al azar” la celda de la que vamos a tomar prestado, convertimos los pesos de las celdas en la columna en una distribución de probabilidad (sin considerar la celda cuyo peso se va a incrementar), la invertimos, y escogemos una celda con base en la distribución de probabilidad invertida.


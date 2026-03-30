# Aplicación sistema de recomendación | descomposición SVD
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

""" 
    Paso 1: Preparar la matriz de datos de preferencias
        1.1. Se carga el documento formato .cvs directamente en mi ruta local con Pandas
        1.2. Limpiamos la tabla de objeto DataFrame que no son escenciales
        1.3. Transformamos el tipo de objeto DataFrame a un tipo puro Numpy
"""
# 1.1. matriz de recomendaciones | datos almacenados en el archivo .csv
archivo = pd.read_csv('C:/Users/ahuer/algebra.lineal.projects/respuestas.csv')

# 1.2.Limpiamos nuestra tabla | matriz de "ruido" o datos incesarios; eliminamos columna de "Marca temporal" (proporcionado por google forms) y fila de recomenadaciones mías.
archivo = archivo.drop(axis=0, index=10, columns='Marca temporal')

# 1.2. En el archivo original .csv todas las respuestas de recomendación de los productos son de tipo Integer, excepto en las que indique como "0 (nunca he ido)" de tipo String
# Convertimos aquellos datos que tengan la particularidad de ser iguales al String indicado en el argumento y le pasamos otro argumento que representa ese 0 y de tipo Integer
R = archivo.replace("0 (nunca he ido)", 0)

# 1.3. El DataFrame o tabla que contiene nuestros datos para que los elementos dentro puedan hacer...
# las operaciones correctamente algebraicamente cambiamos todos nuestro valores de la tabla a double
R = R.astype(dtype=float)

# Imprimos la tabla / DataFrame original que almacena las recomendaciones
print(f"Datos de la encuesta de recomendación usuarios-tiendas ordenados en una tabla:\n{R}\n")

A = np.array(R)

print(f"Calificación de recomendaciones usuarios-tiendas en una matriz:\n {A}\n")

""" 
    Paso 2. Aplicar la descomposición de valores singulares (SVD)
"""
# Llamamos a nuestra funicón de SciPy 'svds()' para pasarle como argumento nuestra matriz a descomponer y especificamos dos ...
# atributos que nos retornarán valores de la descomposición 
U, sigma, Vt = svds(A, k=2, which='LM', return_singular_vectors=True)

print("Matriz descompuesta por svds 'U' (valores propios de R*R-trans):\n",U,"\n\n",
      "vector descompuesto '∑' por svds (valores singulares de la matriz R):\n",sigma,"\n\n",
      "Matriz V-trans (contiene los valores propios de la matriz R-trans * R):\n",Vt,"\n\n")

sigma = np.diag(sigma)
print("Matriz '∑':\n",sigma)

print(f"\n Dimensiones de U={U.shape}, ∑={sigma.shape} y V-transpuesta={Vt.shape}\n")

"""
    Paso 3. Reconstrucción de la matriz de preferencias
"""
R_aprox = U @ sigma @ Vt
print("Matriz reconstruida con recomendaciones a productos que fueron calificados con 0:\n",R_aprox)
print("Elementos de cada dimensión de la matriz R reconstruida:\n",R_aprox.shape,"\n")


"""
    Paso 4. Generar recomendaciones
"""
# Almancenamos el arrglo fila del primer sujeto en un arreglo
sujeto1_R = R_aprox[0]

# Filtro a nuestro arreglo de sujeto1 original
filtro_sujeto1_original = A[0] == 0

# Vector de las predicciones de las tiendas a las que indico "0 (nunca he ido)" del sujeto:0 
prediciciones_nuevas_sujeto1 = R_aprox[0][filtro_sujeto1_original]

# Tomamos los nombres-etiquetas de las tiendas que estan los indices donde hay valores 0 del DataFrame
filtrado_tiendas_sujeto1 = archivo.columns[filtro_sujeto1_original]

# Vector-columna con valores de recomendacion calculados a partir de 'R_aprox' de las tiendas vector-fila que no ha visitado el sujeto:0 
# ...según la respuestas registradas en 'archivo'
recomendaciones_sujeto1 = pd.Series(prediciciones_nuevas_sujeto1, filtrado_tiendas_sujeto1)
# Ordenamos la Tabla final de recomendaciones
recomendaciones_sujeto1 = recomendaciones_sujeto1.sort_values(ascending=False)

print("Tabla de las tiendas mejor recomendadas a las menos recomendadas según la descomposición SVD:")
print(recomendaciones_sujeto1,"\n")

# Mostramos unicamente la tienda que fue la más recomendada
tienda_top1_recomendada = recomendaciones_sujeto1.index[0]
print(f"{tienda_top1_recomendada} es el más recomendado para el sujeto:0")
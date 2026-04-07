""" 
    Código del algoritmo K-means no supervisado en la segmentación de clientes que se sugiere caracterizar por:
    1. Ingresos
    2. Edad
    3. frecuencia de compra
    
    Sin embargo nuestros clusters estarán dados por un Dataset con 3 diferentes enfoques sin embargo muy parecidos y que capturan el mismo objetivo-
"""

# Para esto importamos desde el mismo directorio el archivo .csv que contiene nuestro conjunto de datos a experimentación, además la biblioteca para la manipulación de este.
import pandas as pd
pd.set_option('display.max_columns', None)

# Cargamos nuestro archivo con el dataset .csv
datos = pd.read_csv(filepath_or_buffer='Consumer_Shopping_Trends_2026(2).csv')
#print(f"dataset original en un DataFrame:\n {datos}\n")

# Seleccionamos unicamente nuestras 3 principales caracteristicas que son: edad | ingresos al mes | pedidos en linea al mes
clientes_n = datos.get(['age', 'monthly_income', 'monthly_online_orders'])
#print(f"segmentación del dataset\n{clientes_n}\n")

""" 
    Paso 1; Preprocesar los datos de nuestro dataset.
        Para esto deberemos escalar nuestro datos, dicho en otras palabras vamos a reducir el tamaño gigantesco que puede originarse en los casos en donde el ingreso
        mensual sea mucho mayor a compración de la edad y los pedidos en linea.
        
        Para esto importamos nuestro método StandardScaler del módulo de preprocessing de Scikit-learn
"""
from sklearn.preprocessing import StandardScaler

# Creamos objeto para utilizar todos sus métodos
escalador = StandardScaler()

# Variable que contiene la matriz de los clientes YA ESCALADOS | dentro se guarda el objeto y se llama al método para pasarle como argumento nuestra tabla
escala_clientes = escalador.fit_transform(clientes_n)

#print(f"escalado:\n{escala_clientes}\n")

"""
    Paso 2; Empezamos la primera etapa del algoritmo de K-means.
        Para esta primera etapa debemos llamar a un módulo de la biblioteca llamado 'KMeans' para llamar a su método de 'cluster' para cumplir con la INICIALIZACIÓN.
"""
from sklearn.cluster import KMeans

# Creamos nuestro objeto para utilizar el método de KMeans inicializando la segmentación en 3 clusters | añadimos un argumento que indicará inamobible la aleatoriedad de selección de centroide.
k_means = KMeans(n_clusters=3, random_state=42)

# Llamamos a un método de la clase KMeans que realizara el calculo para determinar los centroidos para los n clusters y que predicirá el indice al que pertence
# cada cliente en los clusters
centroides_clientes = k_means.fit_predict(escala_clientes)
#print(f"indices del cluster al que pertence cada cliente:\n{centroides_clientes}\n")

# Agregar una columna indicando que indice de cluster se encuentra cada cliente de la tabla DataFrame original
datos['Clusters'] = centroides_clientes
#print(datos)

"""
    Aplicación del algoritmo K-means y su grafica de dispersión en 3D.
    Paso 3; Visualización de los ejes 'edad', 'ingreso mensual' y 'pedidos en linea al mes' a partir de su graficación en 3 dimensiones.
"""
import matplotlib.pyplot as plt

# Creamos nuestro objeto plt para crear un gráfico
fig = plt.figure()
# Creamos el lienzo a partir de ese objeto que será 3D
ax = fig.add_subplot(projection='3d')

# Arreglos que contienen los datos de cada cliente de las columnas especificadas y que DataFrame tiene con las respectivas etiquetas
xs = clientes_n['age']
ys = clientes_n['monthly_income']
zs = clientes_n['monthly_online_orders']

# Al lienzo le llamamos la función que graficará cada dato del cliente en un punto de forma que dibujara un gráfico de dispersión 
ax.scatter(xs=xs, ys=ys, zs=zs, c=datos['Clusters'])

# Mostramos lo que representa cada dimensión
ax.set_xlabel('Edad')
ax.set_ylabel('Ingresos mensuales')
ax.set_zlabel('Pedidos en linea al mes')

"""
    Paso EXTRA:
        Analisis de los resultados:
        Sabiendo que cada cliente representa cada fila, tomaremos los indices de clientes (todos aquellos que hayan sido segmentados hacia el indice del cluster 0)
        para poder filtrarlos y enfocarnos en comportamiento de sus datos.
"""

# Variable que guarda la columna 'Clusters' Dataframe con valores de True/False que identifican la segmentación de todos los clientes (será True SI el cliente se segmento al cluster 0)
clientes_cluster0 = datos['Clusters'] == 0
# Variable que guarda una tabla de solo los clientes que hayan sido segmentados al cluster 0
filtro_cluster0 = datos[clientes_cluster0]

#print(clientes_cluster0,"\n")
#print(filtro_cluster0,"\n")

# Sacamos el promedio de los datos de los clientes que estan segmentados en cluster 0 a partir de nuestras 3 caracteristicas de segmentación 
promedio_clientes_cluster0 = filtro_cluster0.get(['age', 'monthly_income', 'monthly_online_orders'])
promedio_clientes_cluster0 = promedio_clientes_cluster0.mean()
print(f"Resultados del promedio de los clientes del cluster 0:\n{promedio_clientes_cluster0}\n")

# Realizamos el mismo filtro para el cluster1 para conocer el promedio de sus datos segmentados
clientes_cluster1 = datos['Clusters'] == 1
# Se crea la tabla con los clientes segmentados al cluster 1
filtro_cluster1 = datos[clientes_cluster1]
# Realizamos las operaciones del promedio para nuestras 3 columnas que representan nuestras 3 caracteristicas de segmentación por lo que debemos constuir la tabla esas columnas
promedio_clientes_cluster1 = filtro_cluster1[['age', 'monthly_income', 'monthly_online_orders']].mean()
print(f"Resultados del promedio de los clientes del cluster 1:\n{promedio_clientes_cluster1}\n")

# Realizamos el mismo filtro para el cluster1 para conocer el promedio de sus datos segmentados
clientes_cluster2 = datos['Clusters'] == 2
# Se crea la tabla con los clientes segmentados al cluster 2
filtro_cluster2 = datos[clientes_cluster2]
# Realizamos las operaciones del promedio para nuestras 3 columnas que representan nuestras 3 caracteristicas de segmentación por lo que debemos constuir la tabla esas columnas
promedio_clientes_cluster2 = filtro_cluster2[['age', 'monthly_income', 'monthly_online_orders']].mean()
print(f"Resultados del promedio de los clientes del cluster 2:\n{promedio_clientes_cluster2}\n")

plt.show()
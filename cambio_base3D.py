import numpy as np

# Vértices de un cubo 
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

# Paso 2: Definir nueva base

theta = np.pi / 4 # 45 grados

v1 = np.array([np.cos(theta), np.sin(theta), 0])
v2 = np.array([-np.sin(theta), np.cos(theta), 0])
v3 = np.array([0, 0, 1])

# Paso 3: Construir matriz P

P = np.column_stack((v1, v2, v3))

# Paso 4: Transformar 
vertices_transformados = vertices @ P
print(f" resultado de la mulplicación de los vértices del cubo con la matriz de transformación:\n{vertices_transformados}\n")

# Inversa de la transformación | La inversa del cambio de base reconstruye la transformación hacía los valores de los vectores originales del objeto.
# Multiplica la inversa de P por los vertices que fueron transformados PERO transpuetos.
inversa_transf_cubo = vertices_transformados @ np.linalg.inv(P)
print(f"vertices reconstruidos por la matriz de los vertices transformados por la inversa del cambio de base:\n{inversa_transf_cubo}")

""" Graficación de los vértices principales originales y los vértices transformados e interpretar los resultados. """

# Importamos la librería que nos servirá a gráficar nuestro objeto y observar las tranformaciones que tengan
import matplotlib.pyplot as plt

# Gráfico del cubo original | Construimos el plano
# 1. Creamos UNA SOLA ventana dividida en dos lienzos 3D (ax1 y ax2)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), subplot_kw={'projection': '3d'})

# 2. Lienzo Izquierdo (ax1): El cubo original (azul) vs el transformado (rojo)

# Utilizamos un técnica conocida como "slicing" que nos permite (en este caso) a guardar en una variable los numeros que se encuentran en cada fila (las filas completas)...
# la columna que queremos en un array, es decír; guardamos los números de las columnas de la matriz formada por los vertices del cubo
columna_x = vertices[:, 0]
columna_y = vertices[:, 1]
columna_z = vertices[:, 2]

# Se crea el cubo y llamamos el método que nos graficara puntos que representarán los vertices y se le pasa como argumento los arrays columna que...
# en el gráfico son las coordenadas de esos vertices
ax1.set_title("BASE ORTONORMAL - CUBO ORIGINAL")
ax1.scatter(columna_x, columna_y, columna_z)

# Estas tres variables nuevas representan los mismo vertices del cubo PERO que han sufrido una transformación, un cambio de base
transf_base_columna_x = vertices_transformados[:, 0]
transf_base_columna_y = vertices_transformados[:, 1]
transf_base_columna_z = vertices_transformados[:, 2]

# Se gráfican vertices del cubo con nueva base y se pintan de color rojo que representan la rotación de 45° contra las manecillas del reloj
ax2.set_title("BASE ORTONORMAL - CUBO ROTADO 45°")
ax2.scatter(transf_base_columna_x, transf_base_columna_y, transf_base_columna_z, c='red')


"""
    El estudiante debera hacer:
        2. Definir otra base no ortogonal (ortonormal también, no?)
"""
v1_1 = np.array([2, 0.5, 1]) # Escalares estaticos que afectan la 'normalidad' de la base y por consiguiente la ortogonalidad 
v2_2 = np.array([-np.sin(theta), np.cos(theta), 0])
v3_3 = np.array([0, 0, 1])

P_1 = np.column_stack((v1_1, v2_2, v3_3)) # Nueva matriz de transformación

vertices_no_ortonormales = vertices @ P_1
ax3.set_title("BASE NO ORTORNORMAL - CUBO DEFORMADO") # NI ORTOGONAL
ax3.scatter(vertices_no_ortonormales[:, 0], vertices_no_ortonormales[:, 1], vertices_no_ortonormales[:, 2], c='green')

plt.show()
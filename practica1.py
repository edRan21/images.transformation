# Implementación de rotación y homotecia usando NumPy, OpenCV y Matplotlib

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar imagen
img = cv2.imread('A.jpg')

# Supongo que significa algo como indentificar la imagen en pixeles y que cada pixel tiene columna y fila que conforman una imagen, entonces son variables globales
rows, cols = img.shape[:2]

# Rotación de 90 grados 
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
rotated_img = cv2.warpAffine(img, M, (cols, rows))

# Rotación de 45 grados
M2 = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated2_img = cv2.warpAffine(img, M2, (cols, rows))

# Homotecia (escala)
scale_matrix = np.array([[2, 0, 0], [0, 2, 0]], dtype=np.float32)
scaled_img = cv2.warpAffine(img, scale_matrix, (cols * 2, rows * 2))

# Escalado de un factor k = 0.5 | Reducción de tamaño a la mitad de la escala original
scale2_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0]], dtype=np.float32)
scaled2_img = cv2.warpAffine(img, scale2_matrix, (cols // 2, rows // 2))   

# Mostrar resultados
plt.subplot(1, 3, 1), plt.imshow(img), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(rotated_img), plt.title('Rotada 90 grados')
plt.subplot(1, 3, 3), plt.imshow(scaled_img), plt.title('Escalada')
plt.subplot(4, 3, 2), plt.imshow(rotated2_img), plt.title('Rotada 45 grados')
plt.subplot(4, 3, 3), plt.imshow(scaled2_img), plt.title('Escalado k = 0.5')
plt.show()
# Implementación de rotación y homotecia usando NumPy, OpenCV y Matplotlib

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar imagen
img = cv2.imread('A.jpg')

# Rotación de 90 grados 
rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
rotated_img = cv2.warpAffine(img, M, (cols, rows))

# Homotecia (escala)
scale_matrix = np.array([[2, 0, 0], [0, 2, 0]], dtype=np.float32)
scaled_img = cv2.warpAffine(img, scale_matrix, (cols * 2, rows * 2))

# Mostrar resultados
plt.subplot(1, 3, 1), plt.imshow(img), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(rotated_img), plt.title('Rotada')
plt.subplot(1, 3, 3), plt.imshow(scaled_img), plt.title('Escalada')
plt.show()
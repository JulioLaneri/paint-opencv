import numpy as np
from scipy.spatial.distance import cdist
import cv2

def ddf_filter(image, window_size, pag1=0.75, pag2=0.25):
    """
    Aplica el Filtro de Distancia Direccional (DDF) a una imagen.

    Argumentos:
    image -- imagen de entrada (matriz NumPy de forma [filas, columnas, canales])
    window_size -- tamaño de la ventana deslizante (int)
    pag1 -- parámetro pag1 para controlar la importancia de la suma de distancias (float, por defecto 0.75)
    pag2 -- parámetro pag2 para controlar la importancia de la suma de ángulos (float, por defecto 0.25)

    Retorna:
    filtered_image -- imagen filtrada (matriz NumPy de forma [filas, columnas, canales])
    """
    rows, cols, channels = image.shape
    filtered_image = np.zeros_like(image)

    # Función para calcular el ángulo entre dos vectores
    def angle(v1, v2):
        v1 = v1.reshape(-1)
        v2 = v2.reshape(-1)
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(dot / norms)

    # Iterar sobre cada píxel de la imagen
    for i in range(window_size // 2, rows - window_size // 2):
        for j in range(window_size // 2, cols - window_size // 2):
            window = image[i - window_size // 2:i + window_size // 2 + 1,
                           j - window_size // 2:j + window_size // 2 + 1]
            window_flat = window.reshape(-1, channels)
            center_pixel = window_flat[window_size ** 2 // 2]

            # Calcular distancias y ángulos entre el píxel central y los demás en la ventana
            distances = cdist([center_pixel], window_flat)[0]
            angles = [angle(center_pixel, p) for p in window_flat]

            # Aplicar la ecuación (6) del DDF
            combined_criteria = (distances ** pag1) * np.prod([np.cos(a) ** pag2 for a in angles], axis=0)
            filtered_pixel = window_flat[np.argmin(combined_criteria)]

            filtered_image[i, j] = filtered_pixel

    return filtered_image

# Cargar una imagen
image = cv2.imread('lena.png')
if image is None:
    raise FileNotFoundError("No se puede leer la imagen, verifica la ruta y el nombre del archivo.")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Agregar ruido gaussiano a la imagen
noise = np.random.normal(0, 30, image.shape)
noisy_image = image + noise.astype(np.int16)

# Aplicar el filtro DDF
window_size = 5  # Tamaño de la ventana deslizante
filtered_image = ddf_filter(noisy_image, window_size, pag1=0.75, pag2=0.25)

# Visualizar las imágenes
cv2.imshow('Original', image)
cv2.imshow('Ruidosa', noisy_image.astype(np.uint8))
cv2.imshow('Filtrada', filtered_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

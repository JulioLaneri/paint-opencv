import numpy as np
import cv2

class ImageProcessingManager():
  """
    Esta libreria contiene la implementación de los procesos internos del
    Editor de Imagenes.
    Desarrollador por: 
  """

  DEFAULT_WIDTH = 512
  DEFAULT_HEIGHT = 512

  def __init__(self):
    super(ImageProcessingManager, self).__init__()

    # Por defecto tenemos una imagen blanca en la pila.
    initial_matrix = np.ones((self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT, 3), np.uint8) * 255
    
    # Estructura de imagenes
    self.stack_images = [initial_matrix]
    
    # Estructura de puntos/lineas
    self.stack_lines = []

  
  def rgb_to_hex(self, rgb):
    """
      Conversor de un string hexadecimal a arreglos.
      Fuente: https://www.codespeedy.com/convert-rgb-to-hex-color-code-in-python/
    """
    return '%02x%02x%02x' % rgb

  #Función que me compartió el compañero Fernando Fabián Brizuela
  def hex_to_rgb(self, hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))

  def last_image(self):
    """
      NO ALTERAR ESTA FUNCION
      Obtenemos la ultima imagen de nuestra estructura.
    """
    return self.stack_images[-1]

  def can_undo(self):
    """
      NO ALTERAR ESTA FUNCION
      Determinamos si la aplicación puede eliminar
      elementos de la pila.
      Debe haber por lo menos más de un elemento para que 
      se pueda deshacer la imagen
    """
    return len(self.stack_images) > 1

  def has_changes(self):
    """
      NO ALTERAR ESTA FUNCION
      Determinamos si la aplicación contiene
      elementos de la pila.
    """
    return len(self.stack_images) > 1

  def add_image(self, image_path):
    """
    Leemos una imagen con OpenCV
    Redimensionamos según los parametros: DEFAULT_WIDTH y DEFAULT_HEIGHT
    Agregamos una nueva imagen redimensionada en la pila.

    Obs: No te olvides de vaciar las colecciones antes de cargar la imagen.
    """
    # Limpiamos las estructuras existentes
    self.stack_images = []
    self.stack_lines = []

    # Cargamos la imagen utilizando OpenCV
    img = cv2.imread(image_path)

    # Convertimos la imagen de BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionamos la imagen al tamaño por defecto del tablero
    resized_img = cv2.resize(img_rgb, (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))

    # Agregamos la imagen redimensionada a la pila de imágenes
    self.stack_images.append(resized_img)

  def save_image(self, filename):
    """
    Guardamos la última imagen
    """
    # Obtenemos la última imagen de la pila de imágenes
    last_image = self.stack_images[-1]

    # Convertimos la imagen de RGB a BGR
    bgr_image = cv2.cvtColor(last_image, cv2.COLOR_RGB2BGR)

    # Guardamos la última imagen en el archivo especificado
    cv2.imwrite(filename, bgr_image)

  def undo_changes(self):
    """
      Eliminamos el ultimo elemento guardado.
    """
    if len(self.stack_images) > 1:
      # Eliminamos la última imagen de la pila de imágenes
      self.stack_images.pop()


  def save_points(self, x1, y1, x2, y2, line_width, color):
    """
      Guardamos informacion de los puntos aqui en self.stack_lines.
    """
    # TU IMPLEMENTACION AQUI
    # Guardamos la información de los puntos
    self.stack_lines.append((x1, y1, x2, y2, line_width, color))

  def add_lines_to_image(self):
    """
    Creamos una matriz, con un conjunto de lineas.
    Estas lineas se obtienen de self.stack_lines.

    Finalmente guardamos a nuestra pila de imagenes: self.stack_images.

    Ayuda: ver documentacion de "cv2.line" para dibujar lineas en una matriz
    Ayuda 2: no se olviden de limpiar self.stack_lines
    Ayuda 3: utilizar el metodo rgb_to_hex para convertir los colores
    """
    # Creamos una copia de la última imagen en la pila de imágenes
    last_image = self.stack_images[-1].copy()

    # Iteramos sobre las líneas en self.stack_lines y las dibujamos en la imagen
    for line in self.stack_lines:
      x1, y1, x2, y2, line_width, color = line
      # Si el color es un código hexadecimal, lo convertimos a una tupla RGB
      if isinstance(color, str) and color.startswith('#'):
        color = self.hex_to_rgb(color)
      # Si el color es una tupla RGB, lo convertimos a hexadecimal
      elif isinstance(color, tuple) and len(color) == 3:
        color = self.rgb_to_hex(color)
      # Convertimos el color a una tupla de enteros
      color = (int(color[0]), int(color[1]), int(color[2]))
      # Convertimos el color a BGR si es una imagen en color
      if len(last_image.shape) == 3:
        color = (color[2], color[1], color[0])
      # Convertimos line_width a entero
      line_width = int(line_width)
      cv2.line(last_image, (x1, y1), (x2, y2), color, line_width)

    # Agregamos la imagen con las líneas dibujadas a la pila de imágenes
    self.stack_images.append(last_image)

    # Limpiamos la estructura de líneas
    self.stack_lines = []

  def black_and_white_image(self):
    """
      Hacemos una copia de la ultima imagen.
      La Convertimos covertimos a blanco y negro.
      Guardamos a la estructura self.stack_images
      Retornamos la imagen procesada.
    """
    # Hacemos una copia de la última imagen en la pila de imágenes
    last_image = self.stack_images[-1].copy()

    # Convertimos la imagen a blanco y negro
    gray_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2GRAY)

    # Convertimos la imagen de gris a BGR
    processed_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Agregamos la imagen procesada a la pila de imágenes
    self.stack_images.append(processed_image)

    # Retornamos la imagen procesada
    return processed_image

  def negative_image(self):
    """
      Hacemos una copia de la ultima imagen.
      Calculamos el negativo de la imagen.
      Guardamos a la estructura self.stack_images
      Retornamos la imagen procesada.
    """

    # Hacemos una copia de la última imagen en la pila de imágenes
    last_image = self.stack_images[-1].copy()

    # Calculamos el negativo de la imagen
    negative_image = 255 - last_image

    # Agregamos la imagen procesada a la pila de imágenes
    self.stack_images.append(negative_image)

    # Retornamos la imagen procesada
    return negative_image

  def global_equalization_image(self):
    """
      Hacemos una copia de la ultima imagen.
      Equalizamos la imagen.
      Guardamos a la estructura self.stack_images
      Retornamos la imagen procesada.
    """

    # Hacemos una copia de la última imagen en la pila de imágenes
    last_image = self.stack_images[-1].copy()

    # Convertimos la imagen al espacio de color YUV
    yuv_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2YUV)

    # Seleccionamos el canal Y
    y_channel = yuv_image[:, :, 0]

    # Equalizamos el canal Y
    equalized_y_channel = cv2.equalizeHist(y_channel)

    # Actualizamos el canal Y en la imagen YUV
    yuv_image[:, :, 0] = equalized_y_channel

    # Convertimos la imagen de YUV a BGR
    processed_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    # Agregamos la imagen procesada a la pila de imágenes
    self.stack_images.append(processed_image)

    # Retornamos la imagen procesada
    return processed_image

  def CLAHE_equalization_image(self, grid=(8, 8), clipLimit=2.0):
    """
      Hacemos una copia de la ultima imagen.
      Equalizamos la imagen usando el algoritmo de CLAHE.
      Guardamos a la estructura self.stack_images
      Retornamos la imagen procesada.
    """
    # Hacemos una copia de la última imagen en la pila de imágenes
    last_image = self.stack_images[-1].copy()

    # Convertimos la imagen al espacio de color YUV
    yuv_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2YUV)

    # Seleccionamos el canal Y (luminosidad)
    y_channel = yuv_image[:, :, 0]

    # Creamos un objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=grid)

    # Aplicamos CLAHE al canal Y
    equalized_y_channel = clahe.apply(y_channel)

    # Actualizamos el canal Y en la imagen YUV
    yuv_image[:, :, 0] = equalized_y_channel

    # Convertimos la imagen de YUV a BGR
    processed_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    # Agregamos la imagen procesada a la pila de imágenes
    self.stack_images.append(processed_image)

    # Retornamos la imagen procesada
    return processed_image

  def contrast_and_brightness_processing_image(self, alpha, beta):
    """
      Hacemos una copia de la ultima imagen.
      Ajustamos la imagen segun parametros alpha y beta.
      Guardamos a la estructura self.stack_images
      Retornamos la imagen procesada.

      Fuente teorica: http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf
      Pagina 103

      OpenCV:
      https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

      Función en OpenCV:
      https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#convertscaleabs
    """
    # Hacemos una copia de la última imagen en la pila de imágenes
    last_image = self.stack_images[-1].copy()

    # Aplicamos la transformación lineal a la imagen
    processed_image = cv2.convertScaleAbs(last_image, alpha=alpha, beta=beta)

    # Agregamos la imagen procesada a la pila de imágenes
    self.stack_images.append(processed_image)

    # Retornamos la imagen procesada
    return processed_image

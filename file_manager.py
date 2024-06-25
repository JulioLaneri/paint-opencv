from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
import tkinter as tk
import cv2
from PIL import Image, ImageTk


class FileManager():
  """
    Libreria de manipulacion de archivos de imagenes.
    Autor: Andres Zorrilla
    Marzo 2020
  """

  @staticmethod
  def upload_image(root, canvas, img_processing_manager):
    root.option_add('*foreground', 'black')
    root.option_add('*activeForeground', 'black')
    root.filename = filedialog.askopenfilename(title = "Seleccionar un Archivo", filetypes = (('Imagen PNG', '.png'), ('Imagen JPG', '.jpg'), ('Todos los archivos', '*')), initialdir=".")
    print (root.filename)

    if img_processing_manager.has_changes():
      if not FileManager.are_you_sure_open_new_file():
        tk.messagebox.showinfo('Sin cambios', 'No se han aplicado cambios.')
        return

    canvas.delete('all')
    img_processing_manager.add_image(root.filename)
    image = img_processing_manager.last_image()
    image =  ImageTk.PhotoImage(image=Image.fromarray(image))
    canvas.add_image(image)
    root.mainloop()
  
  # Reference method https://stackoverflow.com/questions/19476232/save-file-dialog-in-tkinter
  @staticmethod
  def save_image(root, canvas, img_processing_manager):
    root.option_add('*foreground', 'black')
    root.option_add('*activeForeground', 'black')
    ftypes = [('Imagen PNG', '.png'), ('Imagen JPG', '.jpg'), ('Todos los archivos', '*')]
    filename = filedialog.asksaveasfilename(filetypes=ftypes)
    if filename:
      img_processing_manager.save_image(filename)

  @staticmethod
  def quit(root, canvas):
    pass

  @classmethod
  def are_you_sure_open_new_file(cls):
    message_box = tk.messagebox.askquestion ('Abrir nueva imagen', '¿Estas seguro de que quieres abrir este archivo? Se perderan tus cambios anteriores si abres este archivo.', icon = 'warning')
    return message_box == 'yes'

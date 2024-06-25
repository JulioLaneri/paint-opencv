from tkinter import filedialog
from tkinter import messagebox

from tkinter import *
import tkinter as tk

from file_manager import FileManager



class MenuBar(tk.Menu):
  """docstring for MenuBar"""
  def __init__(self, root, canvas, img_processing_manager):
    super(MenuBar, self).__init__()
    self.root = root
    self.canvas = canvas
    self.img_processing_manager = img_processing_manager
    self.init()

  def init(self):
    self.root.config(menu=self)
    filemenu = tk.Menu(self, tearoff=0)
    filemenu.add_command(label="Nuevo")
    filemenu.add_command(label="Abrir", command=self.upload_image)
    filemenu.add_command(label="Guardar", command=self.save_image)
    filemenu.add_command(label="Cerrar")
    filemenu.add_separator()
    filemenu.add_command(label="Salir", command=self.quit)
    editmenu = tk.Menu(self, tearoff=0)
    editmenu.add_command(label="Deshacer")
    helpmenu = tk.Menu(self, tearoff=0)
    helpmenu.add_command(label="Ayuda")
    helpmenu.add_separator()
    helpmenu.add_command(label="Acerca de...")
    self.add_cascade(label="Archivo", menu=filemenu)
    self.add_cascade(label="Editar", menu=editmenu)
    self.add_cascade(label="Ayuda", menu=helpmenu)

  def upload_image(self):
    FileManager.upload_image(self.root, self.canvas, self.img_processing_manager)

  def save_image(self):
    FileManager.save_image(self.root, self.canvas, self.img_processing_manager)

  def quit(self):
    FileManager.quit(self.root, self.canvas)

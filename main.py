"""
Paint Window Manager

In this script, simple Image Processing Application is shown 
to process by certains operations using OpenCV.

Author: Andres Zorrilla
Last modified: March 2020
"""

# from tkinter import Tk, W, E, BOTH, Canvas, W, N, E, S, PhotoImage, Menu, NE, tk.RAISED, SUNKEN, ROUND, TRUE
from tkinter.ttk import Frame, Button, Entry, Style, Label
from tkinter import *
import tkinter as tk
from tkinter.colorchooser import askcolor
from tkinter import filedialog
from tkinter import messagebox
from menu_bar import MenuBar
from menu_paint_buttons import MenuPaintButtonGroup, EnhancementButtonGroup, MenuAction
from image_processing_manager import ImageProcessingManager
from paint_board import PaintBoard

def main():

  root = tk.Tk()
  img_processing_manager = ImageProcessingManager()

  canvas = PaintBoard(root, bg='white', width=512, height=512)

  # Set menu bar
  menubar = MenuBar(root, canvas, img_processing_manager)
  menu_paint_buttons = MenuPaintButtonGroup(root, canvas, img_processing_manager)
  menu_paint_buttons_2 = EnhancementButtonGroup(root, canvas, img_processing_manager)
  menu_action = MenuAction(root, canvas, img_processing_manager)

  menu_action.grid(row=0, column=1, sticky=tk.N)
  menu_paint_buttons.grid(row=1, column=0, sticky=tk.N)
  canvas.grid(row=1, column=1, columnspan=5, sticky=tk.N)
  menu_paint_buttons_2.grid(row=1, column=6, columnspan=5, sticky=tk.N)

  # root.pack(fill=tk.BOTH, expand=True)
  root.mainloop()

if __name__ == '__main__':
  main()
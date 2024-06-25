
import tkinter as tk

class PaintBoard(tk.Canvas):
  def __init__(self, *args, **kwargs):
    super(PaintBoard, self).__init__(*args, **kwargs)
    self.stack = []
    self.temp_lines = []

  def add_image(self, photo):
    img = self.create_image(0, 0, image=photo, anchor=tk.NW)
    self.stack.append([img])

  def add_temp_line(self, temp_line):
    self.temp_lines.append(temp_line)

  def stack_lines(self):
    self.stack.append(self.temp_lines)
    self.temp_lines = []

  def pop(self):
    if len(self.stack) > 0:
      return self.stack.pop()
    else:
      return []
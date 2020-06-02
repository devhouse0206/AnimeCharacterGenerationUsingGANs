from ._anvil_designer import Form1Template
from anvil import *
import anvil.server

class Form1(Form1Template):
  def __init__(self, **properties):
    self.init_components(**properties)

  def button_1_click(self, **event_args):
    result = anvil.server.call('main',self.text_box_1.text,self.text_box_2.text)
    self.image_1.source=result
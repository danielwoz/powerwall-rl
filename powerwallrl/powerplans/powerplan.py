""" Base Powerplan class and registry model. """

# Author: Daniel Williams

__version__ = '0.0.1'


import os
import sys
import platform
from pathlib import Path

MODULE_DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.append(str(Path(MODULE_DIR).parents[1]))

from importlib import import_module

# Traverse the grid cost modules and import them.
for root, dirs, files in os.walk(MODULE_DIR):
    if root == MODULE_DIR:
      continue
    root = root[len(MODULE_DIR):]
    if root.find('pycache') != -1:
      continue
    if len(root) < 1:
      continue

    if (platform.system().startswith('Win')):
      root = root.replace('\\','.')
    else:
      root = root.replace('/','.')

    for file in files:
      if '__init__.py' in file:
        continue
      import_module("." + file[:-3], 'powerwallrl.powerplans' + root)

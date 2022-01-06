"""  Synergy grid and feedback costs. """

# Author: Daniel Williams

__version__ = '0.0.1'

from . import Powerplan

class Default(Powerplan):
  def feedback(self, _):
    return 5.0

  def usage(self, _):
    return 13.7

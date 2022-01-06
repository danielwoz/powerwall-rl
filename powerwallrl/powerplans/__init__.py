"""
Grid electricity costs and feedback rewards.
"""

# Author: Daniel Williams

__version__ = '0.0.1'


registry = {}


def register(cls):
  registry[cls.__name__] = cls  #problem here
  return cls


class PowerplanMetaClass(type):
  def __new__(cls, clsname, bases, attrs):
    newclass = super(PowerplanMetaClass, cls).__new__(cls, clsname, bases,
                                                      attrs)
    register(newclass)
    return newclass


class Powerplan(metaclass=PowerplanMetaClass):
  __metaclass__ = PowerplanMetaClass

  def feedback(self, dt):
    pass

  def usage(self, dt):
    pass

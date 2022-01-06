"""  This module provides settings interface for various parts of powerwall-rl
to operate for an individual home.
"""

# Author: Daniel Williams

__version__ = '0.0.1'

import configparser
import os
from pathlib import Path

import powerwallrl.powerplans
import powerwallrl.powerplans.default
import powerwallrl.powerplans.powerplan


class PowerwallRLConfig(object):

  def __init__(self):
    self.config = configparser.ConfigParser()
    self.config.read(os.path.join(Path.home(), 'powerwall-rl.ini'))

  @property
  def latitude(self):
    return float(self.config['powerwall-rl']['latitude'])

  @property
  def longitude(self):
    return float(self.config['powerwall-rl']['longitude'])

  @property
  def openweathermap_api_key(self):
    return self.config['powerwall-rl']['openweathermap_api_key']

  @property
  def tesla_username(self):
    return self.config['powerwall-rl']['tesla_username']

  @property
  def tesla_password(self):
    return self.config['powerwall-rl']['tesla_password']

  @property
  def local_timezone(self):
    return self.config['powerwall-rl']['local_timezone']

  @property
  def database_location(self):
    if ('database_location' in self.config['powerwall-rl']):
      return str(
        Path(self.config['powerwall-rl']['database_location']).resolve())
    return os.path.join(Path.home(), "powerwall-rl.db")

  @property
  def model_location(self):
    if ('database_location' in self.config['powerwall-rl']):
      return str(
        Path(self.config['powerwall-rl']['model_location']).resolve())
    return os.path.join(Path.home(), "powerwall-model")

  @property
  def grid_plan(self):
    if ('grid_plan' in self.config['powerwall-rl']):
      return  powerwallrl.powerplans.registry[self.config['powerwall-rl']['grid_plan']]()
    return powerwallrl.powerplans.default.Default()

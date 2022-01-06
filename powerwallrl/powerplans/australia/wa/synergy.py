"""  Synergy grid and feedback costs. """

# Author: Daniel Williams

__version__ = '0.0.1'

from powerwallrl.powerplans.australia.wa import WesternAustralia

class Synergy(WesternAustralia):
  pass

class Rebs(Synergy):
  def feedback(self, _):
    return 7.1350

class Debs(Synergy):
  def feedback(self, dt):
    if dt.hour >= 15 and dt.hour < 21:
      return 10.0
    return 2.75

class SmartHome(Synergy):
  def usage(self, dt):
    # Every day offpeak
    if (dt.hour > 21 or dt.hour < 7):
      return 15.3645
    # Weekend Shoulder
    if (dt.weekday() == 5 or dt.weekday() == 6):
      return 29.2100
    # Weekday Shoulder
    if (dt.hour > 7 and dt.hour < 15):
      return 29.2100
    # Weekday Peak
    return 55.7734

class A1(Synergy):
  def usage(self, _):
    return 29.3273

class A1_Debs(A1, Debs):
  pass

class A1_Rebs(A1, Debs):
  pass

class SmartHome_Debs(SmartHome, Debs):
  pass

class SmartHome_Rebs(SmartHome, Debs):
  pass

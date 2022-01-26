""" This module provides a pretty specialized gym for simulating tesla power
battery operations on a home power environment optimizing for reward. Reward is
a product of how much you value grid failure redundancy and how much money you
save from your grid's usage power bill.
"""

# Author: Daniel Williams

__version__ = '0.0.1'

import calendar
import dateutil.tz
import logging
import numpy as np
import random
import sqlite3
from tabulate import tabulate

from datetime import datetime
from datetime import timedelta
from gym import Env
from gym.spaces import Dict, Box
from gym.utils import seeding
from gym.wrappers import FlattenObservation
from pysolar.solar import get_altitude, get_azimuth


class HomePowerEnv(Env):
  def __init__(self, config, powerplan, dayhour_offset=None, debug=True,
               debug_ratio=.001, battery_charge=30,
               randomize_battery_start=True, reward_backup_percent=True,
               reward_battery_left=True):
    # The only action we can set is the target battery charge percentage.
    self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    # Temperature array
    spaces = {
      'uvi': Box(low=0, high=100, shape=(24,), dtype=np.float16),
      'clouds': Box(low=0, high=100, shape=(24,), dtype=np.uint8),
      'temp': Box(low=-100, high=100, shape=(24,), dtype=np.float16),
      #'humidity': Box(low=0, high=100, shape=(24,), dtype=np.uint8),
      'sun_altitude': Box(low=-90, high=90, shape=(24,), dtype=np.float16),
      'sun_azimuth': Box(low=0, high=360, shape=(24,), dtype=np.float16),
      'grid_cost': Box(low=-1, high=1, shape=(24,), dtype=np.float16),
      'hour_of_day': Box(low=0, high=23, shape=(24,), dtype=np.uint8),
      'day_of_week': Box(low=0, high=6, shape=(24,), dtype=np.uint8),
      'battery': Box(low=0, high=100, shape=(1,), dtype=np.uint8),
    }
    # Used to log updates and debugging information.
    self.logger = logging.getLogger()
    self.observation_space = Dict(spaces)
    self.config = config
    self.con = sqlite3.connect(self.config.database_location)
    self.plan = powerplan

    self.dayhour_offset = dayhour_offset
    self.debug = debug
    self.debug_ratio = debug_ratio

    # Total Wh the telsa battery has.
    # TODO(): Get this from tesla API.
    self.battery_capacity = 13500
    # We can only charge or discharge 5KwH of the battery in one hour.
    # TODO(): Add multi-battery support.
    self.max_battery_discharge_rate_ratio = float(
      (5000 / self.battery_capacity) / 1.1)
    self.max_battery_charge_rate_ratio = float(
      (3300 / self.battery_capacity) / 1.1)

    # Set start or randomize battery starting charge on restarts.
    self.battery_charge = battery_charge
    self.randomize_battery_start = randomize_battery_start
    self.reward_backup_percent = reward_backup_percent
    self.reward_battery_left = reward_battery_left

    self.offset = 0
    self.battery_charge_left = 100
    self.seed()
    self.battery_usage = []
    self.battery_charge_list = []
    self.battery_state = []
    self.action_list = []
    self.reward_list = []
    self.after_cost_list = []
    self.default_reward_list = []
    self.datetime_list = []
    self.shortfall_list = []
    self.home_usage_list = []
    self.orig_battery_list = []
    self.solar_list = []
    self.battery_left_list = []
    self.what_to_do = []

  def step(self, action):
    # home_usage = battery_usage + grid + solar
    home_usage = (self.data_set[self.offset][3] +
                  self.data_set[self.offset][2] + self.data_set[self.offset][1])
    # We are short home_usage minus solar.
    short_fall_power = home_usage - self.data_set[self.offset][1]
    # Covert the -1 to 1 back to a battery percentage.
    action = max(0, min(100, round(action[0] * 50 + 50)))

    self.action_list.append(action)
    self.battery_state.append(self.battery_charge)
    self.home_usage_list.append(round(home_usage))
    self.orig_battery_list.append(round(self.data_set[self.offset][2]))
    self.solar_list.append(round(self.data_set[self.offset][1]))

    # Step's current datetime.
    dt = self.dayhour_to_datetime(self.data_set[self.offset][0])

    if short_fall_power > 0:
      default_reward = short_fall_power * self.grid_usage(dt) * -1.0
    else:
      # This will be a negative cost, hence treated as a bass reward.
      default_reward = short_fall_power * self.grid_feedback(dt) * -1.0

    self.default_reward_list.append(round(default_reward))

    reward = float(0.0)

    # Discharge.
    if (action < self.battery_charge and short_fall_power > 0):
      # Find the maximum battery we can use in this step.
      # 1. The short fall, can't use more than we need.
      # 2. We can't use more battery than its max throughput.
      # 3. The available charge percentage setting mulitplied by capacity.
      # Also we can't have negative battery usage.
      battery_usage = min([
        short_fall_power,
        self.max_battery_discharge_rate_ratio * self.battery_capacity,
        (self.battery_charge - action) / 100 * self.battery_capacity
      ])
      self.battery_usage.append(int(battery_usage))

      # Remove our battery usage from the charge percentage.
      self.battery_charge -= round(battery_usage / self.battery_capacity * 100)

      # Assume half the value of the powerwall is the cost savings, therefore the warranted cost per kwh is 23c/2.
      reward -= battery_usage / 1000.0 * 0.115

      # Remove from our short fall power.
      short_fall_power -= battery_usage
      self.what_to_do.append('D')
    # Charging
    elif (action > self.battery_charge and short_fall_power > 0):
      # Find the maximum battery charge we can do in this step.
      # 1. The absolute fastest we can charge.
      # 2. The battery charge percentage increase on our current state.
      # 3. We've charge that battery more than its capacity today. This is
      #    possible but avoided for battery longevity. I'm worried about
      #    our warranty.
      battery_charge = max(
        0,
        min([
          self.max_battery_charge_rate_ratio * self.battery_capacity,
          (action - self.battery_charge) / 100 * self.battery_capacity,
          self.battery_charge_left / 100 * self.battery_capacity
        ]) / 1.1)
      self.battery_usage.append(round(battery_charge * -1))

      # Add the amount we are able to charge this hour.
      if (battery_charge > 0):
        self.battery_charge += round(battery_charge / self.battery_capacity *
                                     100)
        self.battery_charge_left -= round(battery_charge /
                                          self.battery_capacity * 100)

      # Add to our grid short fall power.
      short_fall_power += battery_charge * 1.1
      self.what_to_do.append('C')
    else:
      self.battery_usage.append(0)
      self.what_to_do.append('N')

    self.battery_left_list.append(self.battery_charge_left)

    # It's possible that feedback tarrifs mean this isn't the best behaviour
    # but I don't think the Powerwall will let you feedback to the grid if
    # the battery isn't full. It would be much harder to model if it did. If
    # your feedback tariff was really really high during a solar period the
    # model would learn to fill your battery anyway...
    # 1.1 represents the tax / ineffieciency of lithium batteries.
    if short_fall_power < 0 and self.battery_charge < 100:
      battery_wh_to_full = (
        (100 - self.battery_charge) / 100 * self.battery_capacity) * 1.1
      what = self.what_to_do.pop()
      self.what_to_do.append(what + '+C')
      # Use all our additional electricity to charge the battery.
      if battery_wh_to_full > (short_fall_power * -1):
        self.battery_charge += round(short_fall_power * -1 /
                                     self.battery_capacity / 1.1 * 100)
        self.battery_charge_left -= round(short_fall_power * -1 /
                                          self.battery_capacity / 1.1 * 100)
        self.battery_usage.pop()
        self.battery_usage.append(round(short_fall_power))
        short_fall_power = 0
      # Charge the battery to full
      else:
        self.battery_charge = 100
        self.battery_charge_left -= round(battery_wh_to_full /
                                          self.battery_capacity / 1.1 * 100)
        short_fall_power += battery_wh_to_full
        self.battery_usage.pop()
        self.battery_usage.append(round(battery_wh_to_full * -1))

    # Multiply our shortfall Wh by grid cost.
    if short_fall_power > 0:
      reward -= short_fall_power * self.grid_usage(dt)
    # Multiply additional power by the grid feedback reward.
    elif short_fall_power < 0:
      reward += short_fall_power * self.grid_feedback(dt) * -1.0

    self.after_cost_list.append(int(reward))

    if self.offset == 23 and self.reward_battery_left:
      # Give reward or penalty the value of the battery charge for the next hour
      # compared to initial charge.
      # Maybe this should be the average of the next 3 hours? The max
      # charge/discharge time, but this is probably good enough.
      # All we are trying to do here is not reward the model for always driving
      # initial battery charge to zero.
      reward += self.battery_capacity * (
        (self.battery_charge - self.initial_battery_charge) / 100) * self.grid_usage(dt)

    # Use the no battery scenario cost as the basis for 0 reward.
    reward -= default_reward

    # Value our powerwall as a backup worth 150$ a year and that it's only
    # valuable at that when it is at 65% or more charge.
    # TODO() needs regionalization.
    if self.battery_charge > 65 and self.reward_backup_percent:
      reward += (15000.0 / (360.0 * 24.0))

    self.reward_list.append(int(reward))
    self.datetime_list.append(dt.hour)
    self.shortfall_list.append(int(short_fall_power))

    if (self.offset == 23 and self.debug and random.random() < self.debug_ratio):
      self.logger.info("\n" +
        tabulate([
          ["Battery Percent"] + self.battery_state,
          ["Target Percent"] + self.action_list,
          ["Battery Left"] + self.battery_left_list,
          ["Usage"] + self.home_usage_list,
          ["Solar"] + self.solar_list,
          ["Rewards"] + self.reward_list,
          ["After Cost"] + self.after_cost_list,
          ["Before Cost"] + self.default_reward_list,
          ["Shortfall"] + self.shortfall_list,
          ["Battery WH"] + self.battery_usage,
          ["Mode"] + self.what_to_do,
          ["Orig Battery"] + self.orig_battery_list,
        ],
                 headers=["Metric"] + self.datetime_list) +
        "\nTotal Reward: " + str(sum(self.reward_list)) +
        "\nAfter Cost: " + str(sum(self.after_cost_list) * -1) +
        "\nDefault Cost: " + str(sum(self.default_reward_list) * -1))
    if (self.offset == 23):
      self.battery_state = []
      self.battery_charge_list = []
      self.battery_usage = []
      self.action_list = []
      self.reward_list = []
      self.after_cost_list = []
      self.default_reward_list = []
      self.datetime_list = []
      self.shortfall_list = []
      self.solar_list = []
      self.home_usage_list = []
      self.orig_battery_list = []
      self.battery_left_list = []
      self.what_to_do = []

    self.offset = self.offset + 1

    return self.fill_data(self.offset), reward, (self.offset == 24), {}

  def dayhour_to_datetime(self, dayhour):
    dayhour = str(dayhour)
    return datetime(int(dayhour[0:4]),
                    int(dayhour[4:6]),
                    int(dayhour[6:8]),
                    int(dayhour[8:10]),
                    0,
                    0,
                    tzinfo=dateutil.tz.gettz(self.config.local_timezone))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    self.battery_state = []
    self.battery_charge_list = []
    self.battery_usage = []
    self.action_list = []
    self.reward_list = []
    self.default_reward_list = []
    self.datetime_list = []
    self.shortfall_list = []
    self.solar_list = []
    self.home_usage_list = []
    self.orig_battery_list = []

    # Start with a random amount of battery.
    if self.randomize_battery_start:
      self.battery_charge = self.np_random.randint(0, 100)
    self.initial_battery_charge = self.battery_charge

    if self.dayhour_offset:
      self.data_set = self.get_data(self.dayhour_offset)
      self.dayhour_offset += 24
    else:
      self.data_set = self.get_data()

    self.offset = 0
    self.battery_charge_left = 100

    return self.fill_data(0)

  def fill_data(self, offset):
    inverted_data_set = list(zip(*self.data_set[offset:offset + 24]))
    return_data = {
      'uvi': np.array(inverted_data_set[5], dtype=np.float16),
      'clouds': np.array(inverted_data_set[6], dtype=np.uint8),
      'temp': np.array(inverted_data_set[4], dtype=np.float16),
      #'humidity': np.array(inverted_data_set[7], dtype=np.uint8),
      'sun_altitude': np.array(inverted_data_set[11], dtype=np.float16),
      'sun_azimuth': np.array(inverted_data_set[12], dtype=np.float16),
      'grid_cost': np.array(inverted_data_set[10], dtype=np.float16),
      'hour_of_day': np.array(inverted_data_set[9], dtype=np.uint16),
      'day_of_week': np.array(inverted_data_set[8], dtype=np.uint16),
      'battery': np.array([self.battery_charge], dtype=np.uint8),
    }
    return return_data

  def grid_usage(self, dt):
    """ Return grid cost per Wh. """
    return self.plan.usage(dt) / 1000.0

  def grid_feedback(self, dt):
    """ Return grid reward per Wh for feedback. """
    return self.plan.feedback(dt) / 1000.0

  def render(self):
    pass

  def data_set_size(self):
    if hasattr(self, 'data_set_size_count'):
      return self.data_set_size

    cur = self.con.cursor()
    cur.execute(''' SELECT count(*)
                    FROM powerwall INNER JOIN weather_24
                    ON powerwall.dayhour = weather_24.dayhour ''')
    row_count = cur.fetchall()
    
    # We can only start a episode if we have 48 hours of data into the future,
    # the 24th hour of an episode needs 24 hours of weather forecast
    # observations into the future.
    return row_count[0][0] - 48

  def earliest_datetime(self):
    if hasattr(self, 'earliest_datettime_dt'):
      return self.earliest_datettime_dt

    cur = self.con.cursor()
    cur.execute(''' SELECT powerwall.dayhour
                    FROM powerwall INNER JOIN weather_24
                    ON powerwall.dayhour = weather_24.dayhour
                    ORDER BY powerwall.dayhour
                    LIMIT 1 ''')

    row_count = cur.fetchall()
    return self.dayhour_to_datetime(row_count[0][0])

  def get_data(self, dayhour_offset=None):
    if not dayhour_offset:
      dayhour_offset = self.np_random.randint(0, self.data_set_size())

    earliest_datetime = self.earliest_datetime()
    start_datetime = self.earliest_datetime() + timedelta(hours=dayhour_offset)

    cur = self.con.cursor()
    cur.execute(
      ''' SELECT powerwall.dayhour AS dayhour,
                               powerwall.solar_power AS solar_power,
                               powerwall.battery_power AS battery_power,
                               powerwall.grid_power AS grid_power,
                               weather_24.temp AS temp,
                               weather_24.uvi AS uvi,
                               weather_24.clouds AS clouds,
                               weather_24.humidity AS humidity
          FROM powerwall INNER JOIN weather_24
          ON powerwall.dayhour = weather_24.dayhour
          WHERE powerwall.dayhour > ?
          ORDER BY powerwall.dayhour
          LIMIT 48 ''',
      (str(start_datetime.strftime("%Y%m%d%H")),))

    data = cur.fetchall()
    final_data = []
    for row in data:
      row = list(row)
      current_date = self.dayhour_to_datetime(row[0])
      row.append(current_date.weekday())
      row.append(current_date.hour)
      row.append(float(self.grid_usage(current_date)))
      row.append(
        get_altitude(self.config.latitude, self.config.longitude,
                     current_date + timedelta(minutes=30)))
      row.append(
        get_azimuth(self.config.latitude, self.config.longitude,
                    current_date + timedelta(minutes=30)))
      final_data.append(row)
    return final_data


class HomePowerPredictEnv(HomePowerEnv):
  def __init__(self,
               config,
               powerplan,
               start_datetime=None,
               battery_charge=30,
               debug=True):
    super().__init__(config, powerplan, battery_charge=battery_charge,
                     debug=debug, randomize_battery_start=False)

    self.start_datetime = start_datetime

  def fill_data(self, offset):
    inverted_data_set = list(zip(*self.data_set[offset:offset + 24]))

    return_data = {
      'uvi': np.array(inverted_data_set[2], dtype=np.float16),
      'clouds': np.array(inverted_data_set[3], dtype=np.uint8),
      'temp': np.array(inverted_data_set[1], dtype=np.float16),
      #'humidity': np.array(inverted_data_set[4], dtype=np.uint8),
      'sun_altitude': np.array(inverted_data_set[8], dtype=np.float16),
      'sun_azimuth': np.array(inverted_data_set[9], dtype=np.float16),
      'grid_cost': np.array(inverted_data_set[7], dtype=np.float16),
      'hour_of_day': np.array(inverted_data_set[6], dtype=np.uint16),
      'day_of_week': np.array(inverted_data_set[5], dtype=np.uint16),
      'battery': np.array([self.battery_charge], dtype=np.uint8),
    }

    self.logger.info("Forecast: \n" +
      tabulate([
        ["Day"] + list(map(lambda i : calendar.day_name[i][0:3], inverted_data_set[5])),
        ["Hour"] + list(inverted_data_set[6]),
        ["UVI"] + list(inverted_data_set[2]),
        ["Clouds %"] + list(inverted_data_set[3]),
        ["Temp"] + list(inverted_data_set[1]),
        ["Sun Altitude"] + list(map(lambda i : "%0.2f" % i, inverted_data_set[8])),
        ["Sun Azimuth"] + list(map(lambda i : "%0.2f" % i, inverted_data_set[9])),
        ["Grid Cost"] + list(map(lambda i : "%0.2f" % i, inverted_data_set[7])),
        ["Battery %"] + [self.battery_charge]
      ]))
    return return_data

  def get_data(self, dayhour_offset=None):
    cur = self.con.cursor()
    cur.execute(
      ''' SELECT weather_last.dayhour AS dayhour,
                 weather_last.temp AS temp,
                 weather_last.uvi AS uvi,
                 weather_last.clouds AS clouds,
                 weather_last.humidity AS humidity
          FROM weather_last
          WHERE weather_last.dayhour >= ?
          ORDER BY weather_last.dayhour
          LIMIT 48 ''', (str(self.start_datetime.strftime("%Y%m%d%H")),))
    data = cur.fetchall()

    final_data = []
    for row in data:
      row = list(row)
      current_date = self.dayhour_to_datetime(row[0])
      row.append(current_date.weekday())
      row.append(current_date.hour)
      row.append(float(self.grid_usage(current_date)))
      row.append(
        get_altitude(self.config.latitude, self.config.longitude,
                     current_date + timedelta(minutes=30)))
      row.append(
        get_azimuth(self.config.latitude, self.config.longitude,
                    current_date + timedelta(minutes=30)))
      final_data.append(row)
    return final_data


def MakePowerwallEnv(config, powerplan, **kwargs):
  return FlattenObservation(HomePowerEnv(config, powerplan, **kwargs))


def MakePowerwallPredictEnv(config, powerplan, **kwargs):
  return FlattenObservation(HomePowerPredictEnv(config, powerplan, **kwargs))

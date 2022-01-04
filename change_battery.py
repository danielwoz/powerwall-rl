"""  This script will use the collected weather and power data to create a most
  rewarding model for setting the battery state.
"""

# Author: Daniel Williams

__version__ = '0.0.1'

import logging
import multiprocessing
import os
import sys
import datetime
import math

from powerwallrl.gym.powerwall import MakePowerwallPredictEnv
from powerwallrl.settings import PowerwallRLConfig
from teslapy import Tesla

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
  # Log INFO level message to stdout for the user to see progress.
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  config = PowerwallRLConfig()

  logger.debug("Loading model to determine action. %s",
              config.model_location)

  tesla_api = Tesla(config.tesla_username, verify=True)
  tesla_api.fetch_token()
  battery = tesla_api.battery_list()[0]

  current_percent_charged = math.floor(battery.get_battery_data()['percentage_charged'])
  logger.info("Battery currently has %d%% perent charge.", current_percent_charged)

  current_backup_reserve_percent = math.floor(battery.get_battery_data()['backup']['backup_reserve_percent'])
  logger.info("Battery backup reserve percent is current set to %d%%", current_backup_reserve_percent)

  model = PPO.load(config.model_location)
  env = MakePowerwallPredictEnv(config, battery_charge=current_percent_charged, start_datetime=datetime.datetime.now())
  obs = env.reset()
  action, _states = model.predict(obs, deterministic=True)
  charge_percent = max(0, min(100, round(action[0] * 50 + 50)))

  if charge_percent == current_backup_reserve_percent:
    logger.info("Battery backup reserve percent already set correctly at %d%%", charge_percent)
  else:
    logger.info("Setting battery backup reserve percent to %d%%", charge_percent)
    battery.set_backup_reserve_percent(charge_percent)

  logger.debug("Action taken.")
  del model


if __name__ == "__main__":
  main()

"""  This script will use the collected weather and power data to create a most
  rewarding model for setting the battery state.
"""

# Author: Daniel Williams

__version__ = '0.0.1'

import logging
import multiprocessing
import os
import sys

from powerwallrl.gym.powerwall import HomePowerEnv
from powerwallrl.gym.powerwall import MakePowerwallEnv
from powerwallrl.settings import PowerwallRLConfig

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
  eval_env = FlattenObservation(HomePowerEnv(dayhour_offset=48))
  eval_env = DummyVecEnv([lambda: eval_env])

  logger.debug("Loading model to determine action. %s",
              config.model_location)
  model = PPO.load(config.model_location)

  #    'uvi': Box(low=0, high=100, shape=(24,), dtype=np.float16),
  #    'clouds': Box(low=0, high=100, shape=(24,), dtype=np.uint8),
  #    'temp': Box(low=-100, high=100, shape=(24,), dtype=np.float16),
  #    'sun_altitude': Box(low=-90, high=90, shape=(24,), dtype=np.float16),
  #    'sun_azimuth': Box(low=0, high=360, shape=(24,), dtype=np.float16),
  #    'grid_cost': Box(low=-1, high=1, shape=(24,), dtype=np.float16),
  #    'hour_of_day': Box(low=0, high=23, shape=(24,), dtype=np.uint8),
  #    'day_of_week': Box(low=0, high=6, shape=(24,), dtype=np.uint8),
  #    'battery': Box(low=0, high=100, shape=(1,), dtype=np.uint8),


  action, reward = model.step()


  logger.debug("Action taken.")
  del model


if __name__ == "__main__":
  main()

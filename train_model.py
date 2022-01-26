"""  This script will use the collected weather and power data to create a most
  rewarding model for setting the battery state.
"""

# Author: Daniel Williams

__version__ = '0.0.1'

# To remove GDK errors so that this can run headless.
import matplotlib
matplotlib.use('Agg')

import logging
import multiprocessing
import os
import sys

from powerwallrl.gym.powerwall import HomePowerEnv
from powerwallrl.gym.powerwall import MakePowerwallEnv
from powerwallrl.settings import PowerwallRLConfig

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


def _MakePowerwallEnv():
  config = PowerwallRLConfig()
  return MakePowerwallEnv(config, config.grid_plan, debug=False)


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
  num_cpu = multiprocessing.cpu_count()
  env = SubprocVecEnv([_MakePowerwallEnv for i in range(num_cpu)])

  model = PPO('MlpPolicy', env)

  # TODO(): When the amount of data has doubled we should delete the model and
  # do a safe generate and move / replace.
  if (os.path.exists(config.model_location + ".zip")):
    logger.info("Loading previous model to resume learning. %s",
                config.model_location)
    logger.info("rm %s.zip before training if you want to start afresh",
                config.model_location)
    model.set_parameters(config.model_location)

  eval_env = MakePowerwallEnv(config, config.grid_plan, dayhour_offset=24,
                              randomize_battery_start=False,
                              reward_backup_percent=False,
                              reward_battery_left=False)
  eval_env = DummyVecEnv([lambda: eval_env])

  # Use the entire history as a way to no the real average cost saving.
  max_episodes = int(eval_env.data_set_size() / 24) - 1
  logger.info("Training from data set size / episodes / days: %d ", max_episodes)

  mean_reward, std_reward = evaluate_policy(model,
                                            eval_env,
                                            n_eval_episodes=max_episodes,
                                            warn=False,
                                            render=False,
                                            deterministic=True)
  logger.info("Mean reward before training start: %s", mean_reward)

  i = 0
  while i < 10:
    i += 1
    model.learn(total_timesteps=100000)

    update_env = MakePowerwallEnv(config, config.grid_plan, dayhour_offset=24,
                                  randomize_battery_start=False,
                                  reward_backup_percent=False,
                                  reward_battery_left=False)
    update_env = DummyVecEnv([lambda: update_env])
    mean_reward, std_reward = evaluate_policy(model,
                                              update_env,
                                              n_eval_episodes=max_episodes,
                                              warn=False,
                                              render=False,
                                              deterministic=True)
    model.save(config.model_location)
    if i == 10:
      logger.info("Final mean reward: %s", mean_reward)
    else:
      logger.info("Current mean reward: %s", mean_reward)


  del model


if __name__ == "__main__":
  main()

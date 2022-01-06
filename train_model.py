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
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


def _MakePowerwallEnv():
  config = PowerwallRLConfig()
  return MakePowerwallEnv(config, config.grid_plan)


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

  model = PPO('MlpPolicy', env, verbose=1, device='cpu')

  if (os.path.exists(config.model_location + ".zip")):
    logger.info("Loading previous model to resume learning. %s",
                config.model_location)
    logger.info("rm %s.zip before training if you want to start afresh",
                config.model_location)
    # model.set_parameters(config.model_location)

  eval_env = MakePowerwallEnv(config, config.grid_plan, dayhour_offset=48)
  eval_env = DummyVecEnv([lambda: eval_env])
  mean_reward, std_reward = evaluate_policy(model,
                                            eval_env,
                                            n_eval_episodes=2,
                                            render=False,
                                            deterministic=True)
  logger.info("Mean reward before training: %s", mean_reward)

  model.learn(total_timesteps=25000000)
  model.save(config.model_location)

  eval_env = MakePowerwallEnv(config, config.grid_plan, dayhour_offset=48)
  eval_env = DummyVecEnv([lambda: eval_env])
  mean_reward, std_reward = evaluate_policy(model,
                                            eval_env,
                                            n_eval_episodes=2,
                                            render=False,
                                            deterministic=True)
  logger.info("Mean reward after training: %s", str(mean_reward))
  del model


if __name__ == "__main__":
  main()

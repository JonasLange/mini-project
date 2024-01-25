import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import time
import ray
from ray.tune.registry import register_env
from highway_env import Highway

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import os

def train():
    print("now training")
    config = (
        PPOConfig()
        .environment(env="highway")
        .rollouts(num_rollout_workers=1)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1,evaluation_duration=500,evaluation_duration_unit="timesteps", evaluation_interval=500)
    )
    algo = config.build()
    for _ in range(3):
        algo.train() #retruns progress info
    return algo

def env_creator(env_config):
    env = Highway()
    return env

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
if __name__ == "__main__":
    ray.init()
    register_env("highway",lambda config: ParallelPettingZooEnv(env_creator(config)))
    algo = train()
    print("done training")
    results = algo.evaluate()
    print("----------")
    for key, value in results.items():
        #print(f"Results for {key}: {value}")
        print(f"test")

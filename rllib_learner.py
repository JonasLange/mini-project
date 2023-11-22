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
        .environment(env="Taxi-v3")
        .rollouts(num_rollout_workers=1)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )
    algo = config.build()
    for _ in range(5):
        print(algo.train())
    return algo

def env_creator():
    env = Highway()
    env = ss.dtype_v0(env,int)
    return env


if __name__ == "__main__":
    ray.init()
    register_env("highway",env_creator)
    algo = train()
    print("done training")
    algo.evaluate()
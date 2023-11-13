from pettingzoo import ParallelEnv
import random
from gymnasium import spaces
import functools
import sys, traci
from plexe import Plexe, ACC, CACC, FAKED_CACC, RPM, GEAR, ACCELERATION, SPEED


class Highway(ParallelEnv):
    metadata = {
        "name": "highway_v0",
    }
    def __init__(self):
        self.possible_agents=["vehicle"]
        self.agents = ["vehicle","vehicle","vehicle"]
        self.is_paired_with_runner = False

    def reset(self, seed=None, options=None):
        if not self.is_paired_with_runner:
            self._start()
        else:
            pass

        observations = {}
        for agent in self.agents:
            observations[agent]=random.random()
        infos = {a: {} for a in self.agents}
        return observations, infos
    def step(self, actions):
        observations = {}
        rewards={}
        terminated = {}
        truncated = {}
        infos = {}
        for agent, action in actions.items():
            #if action == "accelerate":
            #    pass
            #elif action == "stay":
            #    pass
            #elif action == "break":
            #    pass
            #else:
            #    raise KeyError("unknow action")
            observations[agent]=random.random()
            rewards[agent]=random.random()
            terminated[agent]=False
            truncated[agent]=False
            infos[agent]=None
        return observations,rewards,terminated,truncated,infos
    def render(self):
        pass
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Discrete(120, start=10)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(3)

    def _start():
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")


#For Testing
from pettingzoo.test import parallel_api_test
if __name__ == "__main__":
    env = Highway()
    parallel_api_test(env, num_cycles=1_000_000)
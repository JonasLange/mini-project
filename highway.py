from pettingzoo import ParallelEnv
import random
from gymnasium import spaces
import functools
import sys, traci, os
from plexe import Plexe, ACC, CACC, FAKED_CACC, RPM, GEAR, ACCELERATION, SPEED
from plexe.vehicle_data import VehicleData
from utils import add_platooning_vehicle, communicate, get_distance, \
    start_sumo, running

run_gui = False

class Highway(ParallelEnv):
    metadata = {
        "name": "highway_v0",
    }
    def __init__(self):
        self.possible_agents=["learner", "vehicle"]
        self.agents = ["learner"]
        self.traci_connected = False

    def reset(self, seed=None, options=None):
        if not self.traci_connected:
            print("first start")
            self._start()
            #traci.close()
            self.traci_connected=True
        else:
            print("resetting existing simulation")
            start_sumo("cfg/freeway.sumo.cfg", True, gui=run_gui)
            random.seed(1)

        # create vehicles and track the learner
        self._add_vehicle(False)
        if run_gui:
            traci.gui.trackVehicle("View #0", "learner")
            traci.gui.setZoom("View #0", 20000)

        observations = {}
        for agent in self.agents:
            observations[agent]=random.random()
        infos = {a: {} for a in self.agents}
        return observations, infos
    def step(self, actions):
        #send the actions
        print(actions)
        for agent, action in actions.items():
            self.plexe.set_cc_desired_speed(agent, action/3.6)
        
        #perform the next simulation step
        traci.simulationStep()

        #ToDo: calc observations/rewards
        observations = {}
        rewards={}
        terminated = {}
        truncated = {}
        infos = {}
        for agent, action in actions.items():
            speed = self.plexe.get_vehicle_data(agent).speed
            observations[agent]=int(speed)
            speed = traci.vehicle.getSpeed(agent)*3.6
            print(f"currently traveling at: {speed} km/h")
            emmissions = traci.vehicle.getCO2Emission(agent)
            print(f"currently emmitting {emmissions} mg CO2 per second")
            rewards[agent]=self._reward(speed,emmissions)
            terminated[agent]=False
            truncated[agent]=False
            infos[agent]=None
        return observations,rewards,terminated,truncated,infos
    def render(self):
        pass
    
    def close(self):
        print("closing now")
        traci.close()


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Discrete(120, start=10)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        #return spaces.Discrete(3)
        return spaces.Discrete(120, start=10)

    def _start(self):
        #if 'SUMO_HOME' in os.environ:
        #    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        #    sys.path.append(tools)
        #else:
        #    sys.exit("please declare environment variable 'SUMO_HOME'")
        random.seed(1)
        start_sumo("cfg/freeway.sumo.cfg", False, gui=run_gui)
        self.plexe = Plexe()
        traci.addStepListener(self.plexe)


    def _add_vehicle(self, real_engine=False):
        """
        Adds the learner to the simulation
        """
        topology = {}

        # add a vehicle that wants to join the platoon
        vid = "learner"
        add_platooning_vehicle(self.plexe, vid, 10, 1, 10, 5, real_engine)
        self.plexe.set_fixed_lane(vid, 1, safe=False)
        traci.vehicle.setSpeedMode(vid, 0)
        self.plexe.set_active_controller(vid, ACC)
        #plexe.set_path_cacc_parameters(vid, distance=JOIN_DISTANCE)
        return topology
    def _reward(self, emmisions, speed):
        return speed/emmisions

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import time
def train(steps = 10_000):
    env = Highway()
    env.reset()
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    model = PPO(MlpPolicy,env,verbose=3,batch_size=256,)
    model.learn(total_timesteps=steps)
    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
def train_ray():
    print("now training")
    config = (
        PPOConfig()
        .environment(env="highway", clip_actions=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="ray_results/" + "highway",
        config=config.to_dict(),
    )


def env_creator():
    env = Highway()
    env = ss.dtype_v0(env,int)
    return env

#For Testing
import ray
from ray.tune.registry import register_env
from pettingzoo.test import parallel_api_test
if __name__ == "__main__":
    ray.init()
    register_env("highway",env_creator)
    train_ray()
    #env = Highway()
    #parallel_api_test(env, num_cycles=500)#testing for 5 seconds each
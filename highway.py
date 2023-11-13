from pettingzoo import ParallelEnv
import random
from gymnasium import spaces
import functools
import sys, traci, os
from plexe import Plexe, ACC, CACC, FAKED_CACC, RPM, GEAR, ACCELERATION, SPEED
from utils import add_platooning_vehicle, communicate, get_distance, \
    start_sumo, running
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
            start_sumo("cfg/freeway.sumo.cfg", True)
            random.seed(1)

        # create vehicles and track the learner
        self._add_vehicle(False)
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
            self.plexe.get_vehicle_data(agent)
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
        #return spaces.Discrete(3)
        return spaces.Discrete(120, start=10)

    def _start(self):
        #if 'SUMO_HOME' in os.environ:
        #    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        #    sys.path.append(tools)
        #else:
        #    sys.exit("please declare environment variable 'SUMO_HOME'")
        random.seed(1)
        start_sumo("cfg/freeway.sumo.cfg", False)
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

#For Testing
from pettingzoo.test import parallel_api_test
if __name__ == "__main__":
    env = Highway()
    parallel_api_test(env, num_cycles=1_000)
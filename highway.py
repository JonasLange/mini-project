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
        self.possible_agents=["vehicle"]
        self.agents = ["vehicle","vehicle","vehicle"]
        self.traci_connected = False

    def reset(self, seed=None, options=None):
        if not self.traci_connected:
            self._start()
            #traci.close()
        else:
            start_sumo("cfg/freeway.sumo.cfg", True)
            random.seed(1)
        #self._start()
        self.traci_connected=True


        observations = {}
        for agent in self.agents:
            observations[agent]=random.random()
        infos = {a: {} for a in self.agents}
        return observations, infos
    def step(self, actions):
        #ToDo: send actions via traci

        #perform the next simulation step
        traci.simulationStep()

        #ToDo: calc observations/rewards
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
        # create vehicles and track the joiner
        self.topology = self._add_vehicle(self.plexe, False)
        traci.gui.trackVehicle("View #0", "learner")
        traci.gui.setZoom("View #0", 20000)

    def _add_vehicle(self, plexe, real_engine=False):
        """
        Adds the learner to the simulation
        """
        topology = {}

        # add a vehicle that wants to join the platoon
        vid = "learner"
        add_platooning_vehicle(plexe, vid, 10, 1, 10, 5, real_engine)
        plexe.set_fixed_lane(vid, 1, safe=False)
        traci.vehicle.setSpeedMode(vid, 0)
        plexe.set_active_controller(vid, ACC)
        #plexe.set_path_cacc_parameters(vid, distance=JOIN_DISTANCE)
        return topology

#For Testing
from pettingzoo.test import parallel_api_test
if __name__ == "__main__":
    env = Highway()
    parallel_api_test(env, num_cycles=1_000)
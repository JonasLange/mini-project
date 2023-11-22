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
        
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.agents:
            self.action_spaces[agent]=self.action_space(agent)
            self.observation_spaces[agent]=self.observation_space(agent)

    def reset(self, seed=None, options=None):
        if not self.traci_connected:
            print("first start")
            self._start()
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
            speed = self.plexe.get_vehicle_data(agent).speed
            observations[agent]=int(speed)
        infos = {a: {} for a in self.agents}
        return observations, infos
    
    def step(self, actions):
        #send the actions
        print(actions)
        for agent, action in actions.items():
            self.plexe.set_cc_desired_speed(agent, 100/3.6)#action/3.6)
        
        #perform the next simulation step
        traci.simulationStep()

        #ToDo: calc observations/rewards
        observations = {}
        rewards={}
        terminated = {}
        truncated = {}
        infos = {}
        for agent, action in actions.items():
            speed = traci.vehicle.getSpeed(agent)*3.6
            observations[agent]=int(speed)
            print(f"currently traveling at: {speed} km/h")
            emmissions = traci.vehicle.getCO2Emission(agent)
            print(f"currently emmitting {emmissions} mg CO2 per second")
            rewards[agent]=self._reward(speed,emmissions)
            terminated[agent]=False
            truncated[agent]=False
            infos[agent]={}
        return observations,rewards,terminated,truncated,infos
    
    def render(self):
        pass
    
    def close(self):
        print("closing now")
        traci.close()


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Discrete(131, start=0)#0-130
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(120, start=10)#10-129

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
        vid = "learner"
        add_platooning_vehicle(self.plexe, vid, 10, 1, 10, 5, real_engine)
        self.plexe.set_fixed_lane(vid, 1, safe=False)
        traci.vehicle.setSpeedMode(vid, 0)
        self.plexe.set_active_controller(vid, ACC)
        return topology
    def _reward(self, emmisions, speed):
        return speed/max(1,emmisions)


from pettingzoo.test import parallel_api_test
if __name__ == "__main__":
    env = Highway()
    parallel_api_test(env, num_cycles=500)#testing for 5 seconds each
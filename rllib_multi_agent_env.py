from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pettingzoo import ParallelEnv
import random
from gymnasium import spaces
import functools
import traci
from plexe import Plexe, ACC, CACC, FAKED_CACC, RPM, GEAR, ACCELERATION, SPEED
from utils import add_platooning_vehicle, communicate, get_distance, \
    start_sumo, running

run_gui = False

class Highway(MultiAgentEnv):
    metadata = {
        "name": "highway_v0",
    }
    def __init__(self, num_learners=1):
        super(Highway,self).__init__()
        self.agents = [str(i) for i in range(num_learners)]
        self._agent_ids = [str(i) for i in range(num_learners)]
        self.traci_connected = False
        

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()
        for agent in self.agents:
            self.action_space[agent]=spaces.Discrete(120, start=10)#10-129 km/h
            self.observation_space[agent]=spaces.Discrete(131, start=0)#0-130 km/h



    def reset(self, seed=None, options=None):
        if not self.traci_connected:
            print("first start")
            self._start()
            self.traci_connected=True
        else:
            print("resetting existing simulation")
            start_sumo("cfg/freeway.sumo.cfg", True, gui=run_gui)

        # create vehicles and track the first learner
        self._add_vehicles()
        if run_gui:
            traci.gui.trackVehicle("View #0", "0")
            traci.gui.setZoom("View #0", 20000)

        observations = {}
        for agent in self.agents:
            speed = self.plexe.get_vehicle_data(agent).speed
            observations[agent]=int(speed)
        infos = {a: {} for a in self.agents}
        print(observations)
        return observations, infos
    
    def step(self, actions):
        #send the actions
        for agent, action in actions.items():
            self.plexe.set_cc_desired_speed(agent, action/3.6)
        
        #perform the next simulation step
        traci.simulationStep()

        #ToDo: calc observations/rewards
        observations = {}
        rewards={}
        terminated = {"__all__":False}
        truncated = {"__all__":False}
        infos = {}
        for agent in self.agents:
            speed = traci.vehicle.getSpeed(agent)*3.6
            observations[agent]=int(speed)
            #print(f"currently traveling at: {speed} km/h")
            emissions = traci.vehicle.getCO2Emission(agent)
            #print(f"currently emmitting {emissions} mg CO2 per second")
            rewards[agent]=self._reward(speed,emissions)
            terminated[agent]=False
            truncated[agent]=False
            infos[agent]={}
        print(f"observations: {observations}")
        print(f"rewards: {rewards}")
        return observations,rewards,terminated,truncated,infos
    
    def render(self):
        pass
    
    def close(self):
        print("closing now")
        traci.close()


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


    def _add_vehicles(self):
        """
        Adds the learner to the simulation
        """
        for vid in self.agents:
            add_platooning_vehicle(self.plexe, vid, 10*(int(vid)+1), 1, 10, 5, False)
            self.plexe.set_fixed_lane(vid, 1, safe=False)
            traci.vehicle.setSpeedMode(vid, 0)
            self.plexe.set_active_controller(vid, ACC)

    def _reward(self, speed, emissions):
        return speed/max(1,emissions)

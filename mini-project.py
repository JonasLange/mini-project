import os
import sys
import random
from utils import add_platooning_vehicle, communicate, get_distance, \
    start_sumo, running

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from plexe import Plexe, ACC, CACC, FAKED_CACC, RPM, GEAR, ACCELERATION, SPEED

# vehicle length
LENGTH = 4



def add_vehicle(plexe, real_engine=False):
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

def main(demo_mode, real_engine, setter=None):
    # used to randomly color the vehicles
    random.seed(1)
    start_sumo("cfg/freeway.sumo.cfg", False)
    plexe = Plexe()
    traci.addStepListener(plexe)
    step = 0
    while running(demo_mode, step, 6000):

        # when reaching 60 seconds, reset the simulation when in demo_mode
        if demo_mode and step == 6000:
            start_sumo("cfg/freeway.sumo.cfg", True)
            step = 0
            random.seed(1)

        traci.simulationStep()

        if step == 0:
            # create vehicles and track the joiner
            topology = add_vehicle(plexe, real_engine)
            traci.gui.trackVehicle("View #0", "learner")
            traci.gui.setZoom("View #0", 20000)
        if step % 10 == 1:
            # simulate vehicle communication every 100 ms
            communicate(plexe, topology)
        step += 1

    traci.close()


if __name__ == "__main__":
    main(True, False)

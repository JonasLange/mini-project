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
    vid = "lerner"
    add_platooning_vehicle(plexe, vid, 10, 1, 10, 5, real_engine)
    plexe.set_fixed_lane(vid, 1, safe=False)
    traci.vehicle.setSpeedMode(vid, 0)
    plexe.set_active_controller(vid, ACC)
    #plexe.set_path_cacc_parameters(vid, distance=JOIN_DISTANCE)
    return topology


def get_in_position(plexe, jid, fid, topology):
    """
    Makes the joining vehicle get close to the join position. This is done by
    changing the topology and setting the leader and the front vehicle for
    the joiner. In addition, we increase the cruising speed and we switch to
    the "fake" CACC, which uses a given GPS distance instead of the radar
    distance to compute the control action
    :param plexe: API instance
    :param jid: id of the joiner
    :param fid: id of the vehicle that will become the predecessor of the joiner
    :param topology: the current platoon topology
    :return: the modified topology
    """
    topology[jid] = {"leader": LEADER, "front": fid}
    plexe.set_cc_desired_speed(jid, SPEED + 15)
    plexe.set_active_controller(jid, FAKED_CACC)
    return topology


def open_gap(plexe, vid, jid, topology, n):
    """
    Makes the vehicle that will be behind the joiner open a gap to let the
    joiner in. This is done by creating a temporary platoon, i.e., setting
    the leader of all vehicles behind to the one that opens the gap and then
    setting the front vehicle of the latter to be the joiner. To properly
    open the gap, the vehicle leaving space switches to the "fake" CACC,
    to consider the GPS distance to the joiner
    :param plexe: API instance
    :param vid: vehicle that should open the gap
    :param jid: id of the joiner
    :param topology: the current platoon topology
    :param n: total number of vehicles currently in the platoon
    :return: the modified topology
    """
    index = int(vid.split(".")[1])
    for i in range(index + 1, n):
        # temporarily change the leader
        topology["v.%d" % i]["leader"] = vid
    # the front vehicle if the vehicle opening the gap is the joiner
    topology[vid]["front"] = jid
    plexe.set_active_controller(vid, FAKED_CACC)
    plexe.set_path_cacc_parameters(vid, distance=JOIN_DISTANCE)
    return topology


def reset_leader(vid, topology, n):
    """
    After the maneuver is completed, the vehicles behind the one that opened
    the gap, reset the leader to the initial one
    :param vid: id of the vehicle that let the joiner in
    :param topology: the current platoon topology
    :param n: total number of vehicles in the platoon (before the joiner)
    :return: the modified topology
    """
    index = int(vid.split(".")[1])
    for i in range(index + 1, n):
        # restore the real leader
        topology["v.%d" % i]["leader"] = LEADER
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

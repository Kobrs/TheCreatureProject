import numpy as np
from neuron import h
from matplotlib import pyplot as plt
from ascii_graph import Pyasciigraph
import time
from ev3 import ev3

import Cells
import ga
import architecture
import brick


port = 'COM5'
# creature_sim_dt = 1./5
sim_time = 200  # [ms]
motor_time = 1000  # NOTE ideally this should = sim_time, but in practice
                   #    sim_time is too small for motors to move

n_cells = architecture.n_cells
sensor_cells = architecture.sensor_cells
trackL_cell = architecture.trackL_cell
trackR_cell = architecture.trackR_cell
gate_cells = architecture.gate_cells
specimen_architecture = architecture.generate_architecture_prebuilt()

noise_mean = 0.01
noise_stdev = 0.1

habbitation_mean_el = 1000

seed = np.random.randint(0,10000)
print "Using the follwoing random generator seed:", seed
np.random.seed(seed)

max_dist = 1024  # TEMP -> depends on lego value range
previous_mean_distances = [0]  # this is an idea to emulate habbitation



def val_map(x, in_min, in_max, out_min, out_max):
    x = float(x)  # make sure that we get float operations
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def map_input(distanceL, distanceR):
    # Dirty, but working way of mapping current
    # Lamp = abs(val_map(distanceL, 100*num_lsources, max_dist, -0.4, -0.04))  # Set last value to 0.01 if you want to allow it to stop completely
    # Ramp = abs(val_map(distanceR, 100*num_lsources, max_dist, -0.4, -0.04))
    # Lamp = abs(val_map(distanceL, 400*num_lsources, max_dist, 0.3, 0.04))  # Set last value to 0.01 if you want to allow it to stop completely
    # Ramp = abs(val_map(distanceR, 400*num_lsources, max_dist, 0.3, 0.04))
    # Lamp = (val_map(distanceL, 400*num_lsources, max_dist, 0.0, 0.35))  # Set last value to 0.01 if you want to allow it to stop completely
    # Ramp = (val_map(distanceR, 400*num_lsources, max_dist, 0.0, 0.35))
    # Lamp = abs(-val_map(distanceL, 100*num_lsources, max_dist, 0.0, 0.35))  # Set last value to 0.01 if you want to allow it to stop completely
    # Ramp = abs(-val_map(distanceR, 100*num_lsources, max_dist, 0.0, 0.35))

    last_mean = sum(previous_mean_distances) / len(previous_mean_distances)
    distanceL -= abs(last_mean)
    distanceR -= abs(last_mean)
    print "Habbituated distances:", distanceL, distanceR
    # FIXME/TODO/NOTE: I've switched Ramp and Lamp places and it looks like working better...
    Ramp = (val_map(distanceL, 0, max(previous_mean_distances)*1.5+0.01, 0.0, 0.35))  # Set last value to 0.01 if you want to allow it to stop completely
    Lamp = (val_map(distanceR, 0, max(previous_mean_distances)*1.5+0.01, 0.0, 0.35))

    # print "gradient of distances:", abs(distanceL - distanceR)
    # distL = distanceL - min([distanceL, distanceR])
    # distR = distanceR - min([distanceL, distanceR])
    # Lamp = (val_map(distL, 0, 300, 0.0, 0.35))  # Set last value to 0.01 if you want to allow it to stop completely
    # Ramp = (val_map(distR, 0, 300, 0.0, 0.35))

    # distL = val_map(distanceL, 100*num_lsources, max_dist, -4, 4) * (-1)
    # distR = val_map(distanceL, 100*num_lsources, max_dist, -4, 4) * (-1)
    # distL = sigmoid(distL)
    # distR = sigmoid(distR)
    # Lamp = val_map(distL, 0, 1, 0, 0.35)  # Set last value to 0.01 if you want to allow it to stop completely
    # Ramp = val_map(distR, 0, 1, 0, 0.35)

    previous_mean_distances.append((distanceL+distanceR) / 2)
    if len(previous_mean_distances) > previous_mean_distances:
        del previous_mean_distances[0]


    return Lamp, Ramp



def get_forces(spikes):
    trackL_spikes = 0
    trackR_spikes = 0
    hist_data = []
    # NOTE: output spikes are reset each iteration, so they always contain
    #       only spikes from previous(current) timestep.
    for src_id, spikes in spikes.iteritems():
        trackL_spikes += len(spikes) if src_id == trackL_cell else 0
        trackR_spikes += len(spikes) if src_id == trackR_cell else 0

        # Now we should make spikes plot
        if len(spikes) != 0:
            hist_data.append(("x=%d id=%d"%(net.cells_dict[src_id].x, src_id), len(spikes)))

    # TODO: CHECK IF THE FORCES AREN'T > 100
    trackL_force = val_map(trackL_spikes, 0, 16, 0, 100)
    trackR_force = val_map(trackR_spikes, 0, 16, 0, 100)

    return trackL_force, trackR_force, hist_data




# Initialize network
print "Initializing network"
interpreter = lambda _: (specimen_architecture, 0, 0)
net = ga.Specimen("", Cells.STDP_Dopamine_Cell, interpreter=interpreter,
                  num_of_regions=1)

net.add_stim_gate_cell(gate_cells[0], [sensor_cells[0]])
net.add_stim_gate_cell(gate_cells[1], [sensor_cells[1]])

# Add gaussian noise to gate cells
net.add_noise_to_cells(gate_cells, mean=noise_mean, stdev=noise_stdev,
					  gate_cell=True)

# Setup spike recording for all cells
net.setup_output_cells(net.cells_dict.keys())

print "Initializing brick communication"
robot = brick.Brick(port=port, motor_time=motor_time)
print "Initializing ascii graph"
ascii_graph = Pyasciigraph()


with ev3.EV3(port_str=port) as lego:
    while True:
        # Run the simulation
        print "Taking care of net feedback"
        print "...waiting for sensory data"
        cmd = robot.get_sensors_state_cmd()
        distanceL, distanceR = cmd.send(lego)
        Lamp, Ramp = map_input(distanceL, distanceR)
        Lgate, Rgate = gate_cells
        net.stimulate_gate_cell(Lgate, Lamp, sim_time)
        net.stimulate_gate_cell(Rgate, Ramp, sim_time)


        spikes = net.get_out_spikes()

        # Stepping robot could be moved to the beginning, run in different thread
        #   to get everything while neuron simulation is running, but check out if
        #   neccessary to do that.
        print "Taking care of robot feedback"
        trackL_force, trackR_force, hist_data = get_forces(spikes)
        print "Applying forces:", trackL_force, trackR_force
        
        trackL_force = int(trackL_force)
        trackL_force = trackL_force if trackL_force < 100 else 100
        trackL_force = trackL_force if trackL_force > 0 else 0
        trackR_force = int(trackR_force)
        trackR_force = trackR_force if trackR_force < 100 else 100
        trackR_force = trackR_force if trackR_force > 0 else 0

        print "...sending command with values:", trackL_force, trackR_force
        cmd = robot.move_tracks_cmd(trackL_force, trackR_force)
        cmd.send(lego)
        time.sleep(0.8) # Give motors some time to move


        # Graph data
        for line in ascii_graph.graph('Spikes in last time window', hist_data):
            print line

        print "Running the simulation step"
        net.sim_init()
        net.run(time=sim_time)
        net.clear_sim()

        # TODO: add clock to make constant time steps
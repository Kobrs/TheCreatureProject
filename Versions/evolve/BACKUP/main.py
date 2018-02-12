import numpy as np
from neuron import h
from matplotlib import pyplot as plt

import time

import Cells
import ga
import Simulation
# import ControlGUI

display_sim = True
creature_sim_dt = 1./20
# engine_force = 0.15
engine_force = 0.4

n_cells = 10
sensor_cells = [1, 2]
trackL_cell = 3
trackR_cell = 4
gate_cells = [-1, -2]

WIDTH = 1920
HEIGHT = 1080
freq_min = 40
freq_max = 120
creature_spawn_loc = [WIDTH/2, 200]
num_lsources = 10
target_respawn_time = 200  # None if not to remove targets
num_creatures = 5


dend_pas = -60.

seed = np.random.randint(0,10000)
print "Using the follwoing random generator seed:", seed
np.random.seed(seed)

max_dist = ((HEIGHT + WIDTH) / 2) * num_lsources * 1.25  # some more or less ok upper value

def dist2freq(dist):
    # Basically turn distance to ligth source to spiking frequency
    # This could be done by applying voltage to additional cell connected to
    #   sensory
    return (dist - 0) * (freq_max - freq_min) / (max_dist - 0) + freq_min

def val_map(x, in_min, in_max, out_min, out_max):
    x = float(x)  # make sure that we get float operations
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def generate_architecture(n, w_min=0, w_max=0.1, d_min=0, d_max=16):
    """
    Generates all to all network architecture with random weights in given range
    """
    architecture = {}
    for cell in xrange(n):
        # Generate all2all connections
        # x = 0
        # x = np.random.randint(0, 256)
        x = 255 if cell in [trackL_cell, trackR_cell] else 0
        conns = []
        for i in xrange(n):
            w = np.random.uniform(low=w_min, high=w_max)
            w_type = np.random.randint(low=0, high=2)
            d = np.random.randint(low=d_min, high=d_max)
            conns.append((i, (w_type, w), d))

        architecture[cell] = {'connections': conns, 'x': x, 'dend_len':100,
                              'dend_pas': dend_pas}
    return architecture


specimen_architecture = generate_architecture(n_cells)

# Connection contains: 'target cell', indicator of conn type(i,e), weight and delay values
# specimen_architecture = {sensor_cells[0]: {'connections': [(trackL_cell, (1, 0.015), 5)],
#                              'x': 0, 'dend_len': 100, 'dend_pas': dend_pas},
#                          sensor_cells[1]: {'connections': [(trackR_cell, (1, 0.03), 5)],
#                              'x': 0, 'dend_len': 100, 'dend_pas': dend_pas},
#                          trackL_cell: {'connections': [],
#                              'x': 0, 'dend_len': 100, 'dend_pas': dend_pas},
#                          trackR_cell: {'connections': [],
#                              'x': 0, 'dend_len': 100, 'dend_pas': dend_pas}}
#



# Prepare the environment
environment = Simulation.Environment(num_of_targets=0,
    num_lsources=num_lsources, WIDTH=WIDTH, HEIGHT=HEIGHT, clock_time=0,
    display_sim=display_sim)


# Spawn creatures
creature_args = {'environment': environment, 'sensor_cells': sensor_cells,
                 'trackL_cells': [trackL_cell], 'trackR_cells':[trackR_cell],
                 'engine_force': engine_force,
                 'creature_spawn_loc': creature_spawn_loc,
                 'motors_min_spikes': 0, 'motors_max_spikes': 6,
                 'smooth_track_control': True, 'use_sensor': False,
                 'logging': True}
for i in xrange(num_creatures):
    x = np.random.randint(int(0.25*WIDTH), int(0.75*WIDTH))
    y = np.random.randint(int(0.25*HEIGHT), int(0.75*HEIGHT))
    creature_args['creature_spawn_loc'] = [x, y]
    environment.spawn_creature(**creature_args)
    environment.creatures[-1].create_from_architecture(specimen_architecture, 0, 0)  # No dopamine
    environment.creatures[-1].add_stim_gate_cell(gate_cells[0], [sensor_cells[0]])
    environment.creatures[-1].add_stim_gate_cell(gate_cells[1], [sensor_cells[1]])




time_counter = 0
while True:
    print "stepping"
    environment.step_sim()

    for i, creature in enumerate(environment.creatures):
        # Now check distance to ligth source
        distanceL = creature.body.light_distL
        distanceR = creature.body.light_distR
        """
        Lcell_delay = 1./dist2freq(distanceL)
        Rcell_delay = 1./dist2freq(distanceR)

        # NOTE: Neuron time resets each creature timestep, so creature_sim_dt is
        #       highest occuring neuron time value, but in ms.
        # Create stimulation dictionary
        Lnum_spikes = np.array(np.random.normal(creature_sim_dt/Lcell_delay, 0.6), dtype=int)
        Rnum_spikes = np.array(np.random.normal(creature_sim_dt/Rcell_delay, 0.6), dtype=int)
        # Lnum_spikes = int(creature_sim_dt/Lcell_delay)
        # Rnum_spikes = int(creature_sim_dt/Rcell_delay)
        stim_dict = {sensor_cells[0]: [Lcell_delay*i for i in xrange(Lnum_spikes)],
                     sensor_cells[1]: [Rcell_delay*i for i in xrange(Rnum_spikes)]}

        print "lens:", [len(stim_dict[sensor_cells[i]]) for i in xrange(2)]

        creature.apply_stimuli(stim_dict)
        """

        # Dirty, but working way of mapping current
        Lamp = abs(val_map(distanceL, 0, max_dist, -0.4, -0.04))  # Set last value to 0.01 if you want to allow it to stop completely
        Ramp = abs(val_map(distanceR, 0, max_dist, -0.4, -0.04))
        creature.stimulate_gate_cell(gate_cells[0], Lamp, creature.neuron2creature_dt)
        creature.stimulate_gate_cell(gate_cells[1], Ramp, creature.neuron2creature_dt)

        if target_respawn_time is not None:
            if time_counter % target_respawn_time == 0:
                # Remove and respawn one target
                creature.remove_light_source(np.random.randint(0, num_lsources))
                creature.spawn_light_sources(1)


        # # Release dopamine if picked up something
        # if creature.body.picked_up:
        #     print "----------------------------------------RELEASING DOPAMINE!"
        #     region = np.random.randint(0, creature.num_of_regions)
        #     creature.stimulate_pleasure(region, dop=100, dopdr=0.005)

        # Release dopamine only on  motory cells
        # if creature.body.picked_up:
        #     print "----------------------------------------RELEASING DOPAMINE!"
        #     region = creature.


    time_counter += 1

# NOTE: IT *MAY* BE WORKING, BUT SOMETHING'S UP WITH THE CODE, AS TIMESTEP
#       TAKES LOGNGER AND LONGER TO COMPUTE AS TIME GOES BY.
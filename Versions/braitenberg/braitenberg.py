import numpy as np
from neuron import h
from matplotlib import pyplot as plt

import time

import Cells
import ga
import Creature
# import ControlGUI

display_sim = True
creature_sim_dt = 1./20
engine_force = 0.1

sensor_cells = [1, 2]
trackL_cell = 3
trackR_cell = 4
gate_cells = [-1, -2]

WIDTH = 1920
HEIGHT = 1080
freq_min = 40
freq_max = 120
creature_spawn_loc = [WIDTH/2, 200]
# light_sources = [[100, 700]]
light_sources = [(np.random.randint(low=0, high=WIDTH), np.random.randint(low=0, high=HEIGHT)) for _ in xrange(10)]


dend_pas = -60.

seed = np.random.randint(0,10000)
print "Using the follwoing random generator seed:", seed
np.random.seed(seed)

# max_dist = (HEIGHT**2 + WIDTH**2)**0.5
# max_dist = (HEIGHT**2 + WIDTH**2)**0.5 / 2.  # TEMP
max_dist = ((HEIGHT + WIDTH) / 2) * len(light_sources)

def dist2freq(dist):
    # Basically turn distance to ligth source to spiking frequency
    # This could be done by applying voltage to additional cell connected to
    #   sensory
    return (dist - 0) * (freq_max - freq_min) / (max_dist - 0) + freq_min

def val_map(x, in_min, in_max, out_min, out_max):
    x = float(x)  # make sure that we get float operations
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# Connection contains: 'target cell', indicator of conn type(i,e), weight and delay values
specimen_architecture = {sensor_cells[0]: {'connections': [(trackL_cell, (1, 0.015), 5)],
                             'x': 0, 'dend_len': 100, 'dend_pas': dend_pas},
                         sensor_cells[1]: {'connections': [(trackR_cell, (1, 0.03), 5)],
                             'x': 0, 'dend_len': 100, 'dend_pas': dend_pas},
                         trackL_cell: {'connections': [],
                             'x': 0, 'dend_len': 100, 'dend_pas': dend_pas},
                         trackR_cell: {'connections': [],
                             'x': 0, 'dend_len': 100, 'dend_pas': dend_pas}}

creature = Creature.Creature(display_sim=display_sim, clock=0,
    creature_sim_dt=creature_sim_dt, sensor_cells=sensor_cells,
    trackL_cells=[trackL_cell], trackR_cells=[trackR_cell],
    creature_lifetime=100000, num_of_targets=0, light_sources=light_sources,
    creature_spawn_loc=creature_spawn_loc, WIDTH=WIDTH, HEIGHT=HEIGHT,
    smooth_track_control=True, motors_min_spikes=0, motors_max_spikes=6,
    logging=True)

creature.create_from_architecture(specimen_architecture, 0, 0)  # No dopamine
creature.add_stim_gate_cell(gate_cells[0], [sensor_cells[0]])
creature.add_stim_gate_cell(gate_cells[1], [sensor_cells[1]])


# TODO: Now the problem is that distance betweeen detecting cells is too small
#       for frequency encoding to create any differences in activity of
#       both sensory cells. This can be fixed either by increasing sensor
#       distances, or by introducing field of view, where if creatue is facing
#       its left side to the light source, the right side is not activated at all.
#       Other option is to change the probability of number of spikes, depending
#       on distance, not round the values to always get consistent values.
# NOTE: testing third optinon

counter = 0
while True:
    c_x, c_y = creature.get_position()
    # NOTE uncomment if creature should die when touching borders
    # while (creature.get_position()[0] > 0 and creature.get_position()[0] < WIDTH
    #         and creature.get_position()[1] > 0 and creature.get_position()[1] < HEIGHT):
    while True:
        print "stepping"
        creature.step()

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

        counter += 1
    print counter

    # NOTE
    break

    # When creature went 'off the road', we have to reposition it, but without
    #   resetting creature object
    # print "SETTING POSITION"
    creature.set_poistion(*creature_spawn_loc)



# NOTE: IT *MAY* BE WORKING, BUT SOMETHING'S UP WITH THE CODE, AS TIMESTEP
#       TAKES LOGNGER AND LONGER TO COMPUTE AS TIME GOES BY.
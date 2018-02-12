import numpy as np
from neuron import h
from matplotlib import pyplot as plt
import pickle

import time

import Cells
import ga
import Simulation
import architecture
# import ControlGUI

display_sim = True
creature_sim_dt = 1./5
motor_max_spikes = 5
# engine_force = 0.15
engine_force = 0.2

n_cells = architecture.n_cells
sensor_cells = architecture.sensor_cells
trackL_cell = architecture.trackL_cell
trackR_cell = architecture.trackR_cell
gate_cells = architecture.gate_cells

WIDTH = 1920
HEIGHT = 1440
freq_min = 40
freq_max = 120
creature_spawn_loc = [WIDTH/2, 200]
num_lsources = 1
target_respawn_time = 10000  # None if not to remove targets
noise_mean = 0.01
noise_stdev = 0.1

num_creatures = 1
# score_decay = 1./(20 * 20) 
score_decay = 0
# init_score = 1.5
init_score = 0
copy_score =  4000

score_after_time = 2000
habbitation_mean_el = 100



seed = np.random.randint(0,10000)
print "Using the follwoing random generator seed:", seed
np.random.seed(seed)

max_dist = ((HEIGHT + WIDTH) / 2) * num_lsources * 0.5  # some more or less ok upper value
previous_mean_distances = [0]  # this is an idea to emulate habbitation

def dist2freq(dist):
    # Basically turn distance to ligth source to spiking frequency
    # This could be done by applying voltage to additional cell connected to
    #   sensory
    return (dist - 0) * (freq_max - freq_min) / (max_dist - 0) + freq_min


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

    # Ramp = abs(-(val_map(distanceL, 0, max_dist, 0.05, 0.35)))  # Set last value to 0.01 if you want to allow it to stop completely
    # Lamp = abs(-(val_map(distanceR, 0, max_dist, 0.05, 0.35)))
    # Lamp = (val_map(distanceL, 0, max_dist, 0.05, 0.35))  # Set last value to 0.01 if you want to allow it to stop completely
    # Ramp = (val_map(distanceR, 0, max_dist, 0.05, 0.35))

    # Lamp = abs(-val_map(distanceL, 100*num_lsources, max_dist, 0.0, 0.35))  # Set last value to 0.01 if you want to allow it to stop completely
    # Ramp = abs(-val_map(distanceR, 100*num_lsources, max_dist, 0.0, 0.35))

    last_mean = sum(previous_mean_distances) / len(previous_mean_distances)
    distanceL -= last_mean
    distanceR -= last_mean
    # print "Habbituated distances:", distanceL, distanceR, last_mean
    # # FIXME/TODO/NOTE: I've switched Ramp and Lamp places and it looks like working better...
    # # Ramp = (val_map(distanceL, 0, max(previous_mean_distances)*1.5+0.01, 0.0, 0.35))  # Set last value to 0.01 if you want to allow it to stop completely
    # # Lamp = (val_map(distanceR, 0, max(previous_mean_distances)*1.5+0.01, 0.0, 0.35))

    val_range = max(previous_mean_distances)-min(previous_mean_distances)*1.25 +0.1
    distL = val_map(distanceL, -0.25*val_range, val_range, -4, 4)
    distR = val_map(distanceR, -0.25*val_range, val_range, -4, 4)
    distL = sigmoid(distL)
    distR = sigmoid(distR)
    # NOTE: sensors are exchanged
    Ramp = val_map(distL, 0, 1, 0, 0.35)  # Set last value to 0.01 if you want to allow it to stop completely
    Lamp = val_map(distR, 0, 1, 0, 0.35)

    # val_range = max(previous_mean_distances)-min(previous_mean_distances)*1.25 +0.1
    # distL = abs(-val_map(distanceL, -0.25*val_range, val_range, -4, 4))
    # distR = abs(-val_map(distanceR, -0.25*val_range, val_range, -4, 4))
    # distL = sigmoid(distL)
    # distR = sigmoid(distR)
    # # NOTE: sensors are exchanged
    # Lamp = val_map(distL, 0, 1, 0.05, 0.35)  # Set last value to 0.01 if you want to allow it to stop completely
    # Ramp = val_map(distR, 0, 1, 0.05, 0.35)


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


# x = np.linspace(0, max_dist, 1000)
# f1 = lambda distanceL: val_map(distanceL, 0, max_dist, -4, 4) * (-1)
# f2 = lambda distL: sigmoid(distL)
# f3 = lambda distL: val_map(distL, 0, 1, 0, 0.35)
# y1 = map(f1, x)
# y2 = map(f2, y1)
# y3 = map(f3, y2)
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.plot(x, map(lambda x: map_input(x, x)[0], x))
# plt.show()

# raise SystemExit





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
                 'motors_min_spikes': 0, 'motors_max_spikes': motor_max_spikes,
                 'smooth_track_control': True, 'use_sensor': False,
                 'logging': True}
for i in xrange(num_creatures):
    specimen_architecture = architecture.generate_architecture_prebuilt(n_cells)
    # specimen_architecture = architecture.generate_architecture_braitenberg()

    x = np.random.randint(int(0.25*WIDTH), int(0.75*WIDTH))
    y = np.random.randint(int(0.25*HEIGHT), int(0.75*HEIGHT))
    creature_args['creature_spawn_loc'] = [x, y]
    environment.spawn_creature(**creature_args)
    environment.creatures[-1].create_from_architecture(specimen_architecture, 0, 0)  # No dopamine
    environment.creatures[-1].add_stim_gate_cell(gate_cells[0], [sensor_cells[0]])
    environment.creatures[-1].add_stim_gate_cell(gate_cells[1], [sensor_cells[1]])
    # Add gaussian noise to gate cells
    environment.creatures[-1].add_noise_to_cells(gate_cells, mean=noise_mean,
                                                 stdev=noise_stdev, gate_cell=True)
    environment.creatures[-1].body.score = init_score




time_counter = 0
while True:
    print "stepping"
    environment.step_sim()

    for creature_i, creature in enumerate(environment.creatures):
        print "Stepping cell with index=", i
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


        print "Sum of distances to targets from sensors:", distanceL, distanceR
        Lamp, Ramp = map_input(distanceL, distanceR)
        Lgate, Rgate = gate_cells
        creature.stimulate_gate_cell(Lgate, Lamp, creature.neuron2creature_dt)
        creature.stimulate_gate_cell(Rgate, Ramp, creature.neuron2creature_dt)



        # # Release dopamine if picked up something
        # if creature.body.picked_up:
        #     print "----------------------------------------RELEASING DOPAMINE!"
        #     region = np.random.randint(0, creature.num_of_regions)
        #     creature.stimulate_pleasure(region, dop=100, dopdr=0.005)

        # Release dopamine only on  motory cells
        # if creature.body.picked_up:
        #     print "----------------------------------------RELEASING DOPAMINE!"
        #     region = creature.


        # Check score (food level), copy and reset creature if exceeds some food level,
        # kill if below 0
        if creature.body.score > copy_score:
            # Copy and reset creature
            architecture = creature.architecture
            environment.spawn_creature(**creature_args)
            environment.creatures[-1].create_from_architecture(architecture, 0, 0)
            environment.creatures[-1].add_stim_gate_cell(gate_cells[0], [sensor_cells[0]])
            environment.creatures[-1].add_stim_gate_cell(gate_cells[1], [sensor_cells[1]])
            environment.creatures[-1].body.score = init_score
            creature.body.score = init_score

        if creature.body.score < 0:
            # Kill creature
            environment.kill_creature(creature_i)


        creature.body.score -= score_decay
        print "Creature %d score=%.2f"%(creature_i, creature.body.score)


    if target_respawn_time is not None:
        if time_counter % target_respawn_time == 0:
            # Remove and respawn one target
            environment.remove_light_source(np.random.randint(0, len(environment.light_sources)))
            environment.spawn_light_sources(1)

    # Maintain constant amount fo light sources (food)
    print "spawning %d light sources"%(num_lsources - len(environment.light_sources))
    environment.spawn_light_sources(num_lsources - len(environment.light_sources))


    print "@@@@ CREATURE TIME: %d @@@@"%time_counter

    if time_counter%1000 == 0:
        architectures = []
        for creature in environment.creatures:
            architectures.append(creature.architecture)
        with open("models/architectures%d.pickle"%time_counter, 'w') as f:
            pickle.dump(architectures, f)


    if score_after_time is not None and score_after_time == time_counter:
        scores = [creature.body.score for creature in environment.creatures]
        print "\n\nScore after %d timesteps: %d\n"%(time_counter, max(scores))
        break

    time_counter += 1

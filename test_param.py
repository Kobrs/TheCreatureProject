"""
Implementation based on main.py, but with keyword arguments to define params
and disabled logging ecept from final result
"""

import numpy as np
from neuron import h
from matplotlib import pyplot as plt
import pickle
import time
import argparse

import Cells
import ga
import Simulation
import architecture


parser = argparse.ArgumentParser()
parser.add_argument('--engine_force', '-v', type=float, default=0.2)
parser.add_argument('--noise_mean', type=float, default=0.01)
parser.add_argument('--noise_stdev', type=float, default=0.1)
parser.add_argument('--type', type=str)


display_sim = False
creature_sim_dt = 1./5
motor_max_spikes = 6
# engine_force = 0.15
engine_force = 0.2

n_cells = architecture.n_cells
sensor_cells = architecture.sensor_cells
trackL_cell = architecture.trackL_cell
trackR_cell = architecture.trackR_cell
gate_cells = architecture.gate_cells

WIDTH = 1920
HEIGHT = 1440
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

habbitation_mean_el = 200

# seed = np.random.randint(0,10000)
# print "Using the follwoing random generator seed:", seed
# np.random.seed(seed)

previous_mean_distances = [0]  # this is an idea to emulate habbitation



def val_map(x, in_min, in_max, out_min, out_max):
    x = float(x)  # make sure that we get float operations
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def map_input(distanceL, distanceR):
    last_mean = sum(previous_mean_distances) / len(previous_mean_distances)
    distanceL -= last_mean
    distanceR -= last_mean
    # print "Habbituated distances:", distanceL, distanceR, last_mean
    # FIXME/TODO/NOTE: I've switched Ramp and Lamp places and it looks like working better...
    # Ramp = (val_map(distanceL, 0, max(previous_mean_distances)*1.5+0.01, 0.0, 0.35))  # Set last value to 0.01 if you want to allow it to stop completely
    # Lamp = (val_map(distanceR, 0, max(previous_mean_distances)*1.5+0.01, 0.0, 0.35))

    val_range = max(previous_mean_distances)-min(previous_mean_distances)*1.25 +0.1
    distL = val_map(distanceL, -0.25*val_range, val_range, -4, 4)
    distR = val_map(distanceR, -0.25*val_range, val_range, -4, 4)
    distL = sigmoid(distL)
    distR = sigmoid(distR)
    # NOTE: sensors are exchanged
    Ramp = val_map(distL, 0, 1, 0, 0.35)  # Set last value to 0.01 if you want to allow it to stop completely
    Lamp = val_map(distR, 0, 1, 0, 0.35)


    previous_mean_distances.append((distanceL+distanceR) / 2)
    if len(previous_mean_distances) > previous_mean_distances:
        del previous_mean_distances[0]


    return Lamp, Ramp



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
    # print "stepping"
    environment.step_sim()

    for creature_i, creature in enumerate(environment.creatures):
        # print "Stepping cell with index=", i
        # Now check distance to ligth source
        distanceL = creature.body.light_distL
        distanceR = creature.body.light_distR


        # print "Sum of distances to targets from sensors:", distanceL, distanceR
        Lamp, Ramp = map_input(distanceL, distanceR)
        Lgate, Rgate = gate_cells
        creature.stimulate_gate_cell(Lgate, Lamp, creature.neuron2creature_dt)
        creature.stimulate_gate_cell(Rgate, Ramp, creature.neuron2creature_dt)



        # # Release dopamine if picked up something
        # if creature.body.picked_up:
            # print "----------------------------------------RELEASING DOPAMINE!"
        #     region = np.random.randint(0, creature.num_of_regions)
        #     creature.stimulate_pleasure(region, dop=100, dopdr=0.005)

        # Release dopamine only on  motory cells
        # if creature.body.picked_up:
            # print "----------------------------------------RELEASING DOPAMINE!"
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
        # print "Creature %d score=%.2f"%(creature_i, creature.body.score)


    if target_respawn_time is not None:
        if time_counter % target_respawn_time == 0:
            # Remove and respawn one target
            environment.remove_light_source(np.random.randint(0, len(environment.light_sources)))
            environment.spawn_light_sources(1)

    # Maintain constant amount fo light sources (food)
    # print "spawning %d light sources"%(num_lsources - len(environment.light_sources))
    environment.spawn_light_sources(num_lsources - len(environment.light_sources))


    # print "@@@@ CREATURE TIME: %d @@@@"%time_counter

    if time_counter%1000 == 0:
        architectures = []
        for creature in environment.creatures:
            architectures.append(creature.architecture)
        with open("models/architectures%d.pickle"%time_counter, 'w') as f:
            pickle.dump(architectures, f)


    if score_after_time is not None and score_after_time == time_counter:
        scores = [creature.body.score for creature in environment.creatures]
        # print "\n\nScore after %d timesteps: %d\n"%(time_counter, max(scores))
        break

    time_counter += 1

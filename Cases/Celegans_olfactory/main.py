import numpy as np
from neuron import h
from matplotlib import pyplot as plt

import time

import Cells
import ga
import Simulation
import architecture
import argparse

# Add arguments - they will be used f
parser = argparse.ArgumentParser()
parser.add_argument("--architecture", type=str, default="prebuilt")
parser.add_argument("--noise_mean", type=float, default=None)
parser.add_argument("--noise_stdev", type=float, default=None)
parser.add_argument("--lsources", type=int, default=None)
parser.add_argument("--score_time", type=int, default=None)
parser.add_argument("--no_STDP", action='store_true')
parser.add_argument("--not_display", action='store_true')
parser.add_argument("--silent", action='store_true')
parser.add_argument("--frozen", action='store_true')
ag = parser.parse_args()
logging = not ag.silent

display_sim = True
creature_sim_dt = 1./20
dt_scale_factor = creature_sim_dt / 1./20

motor_max_spikes = 6*dt_scale_factor
# engine_force = 0.15
engine_force = 0.2*dt_scale_factor

n_cells = architecture.n_cells
sensor_cells = architecture.sensor_cells
trackL_cell = architecture.trackL_cell
trackR_cell = architecture.trackR_cell
gate_cells = architecture.gate_cells

WIDTH = 1920
HEIGHT = 1080
creature_spawn_loc = [WIDTH/2, 200]
num_lsources = 1
target_respawn_time = 10000  # None if not to remove targets
noise_mean = 0.01
noise_stdev = 0.1

score_after_time = 1e9
habbitation_mean_el = 100

# Commandline arguments are always superior to default settings
if ag.noise_mean is not None: noise_mean = ag.noise_mean
if ag.noise_stdev is not None: noise_stdev = ag.noise_stdev
if ag.lsources is not None: num_lsources = ag.lsources
if ag.score_time is not None: score_after_time = ag.score_time
if ag.not_display is True: display_sim = False


seed = np.random.randint(0,10000)
print "Using the follwoing random generator seed:", seed
np.random.seed(seed)

max_dist = ((HEIGHT + WIDTH) / 2) * num_lsources * 0.5  # some more or less ok upper value
previous_mean_distances = [0]  # this is an idea to emulate habbitation



def log(*args):
    if logging:
        for arg in args:
            print arg
        print ""


def val_map(x, in_min, in_max, out_min, out_max):
    x = float(x)  # make sure that we get float operations
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def map_input(distanceL, distanceR):

    last_mean = sum(previous_mean_distances) / len(previous_mean_distances)
    distanceL -= last_mean
    distanceR -= last_mean

    val_range = max(previous_mean_distances)-min(previous_mean_distances)*1.25 +0.1
    distL = val_map(distanceL, -0.125*val_range, val_range, -4, 4)
    distR = val_map(distanceR, -0.125*val_range, val_range, -4, 4)
    distL = sigmoid(distL)
    distR = sigmoid(distR)
    Lamp = val_map(distL, 0, 1, 0.0, 0.35)  # Set last value to 0.01 if you want to allow it to stop completely
    Ramp = val_map(distR, 0, 1, 0.0, 0.35)

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
# p1 = plt.plot(x, y1)
# p2 = plt.plot(x, y2)
# p3 = plt.plot(x, y3)
# p4 = plt.plot(x, map(lambda x: map_input(x, x)[0], x))
# plt.legend(p1+p2+p3+p4, ['p1', 'p2', 'p3', 'p4'])
# plt.show()

# raise SystemExit


# Prepare the environment
environment = Simulation.Environment(num_of_targets=0,
    num_lsources=num_lsources, WIDTH=WIDTH, HEIGHT=HEIGHT, clock_time=0,
    creature_sim_dt=creature_sim_dt, display_sim=display_sim, logging=logging)
environment.frozen = ag.frozen


# Spawn creatures
creature_args = {'environment': environment, 'sensor_cells': sensor_cells,
                 'trackL_cells': [trackL_cell], 'trackR_cells':[trackR_cell],
                 'engine_force': engine_force,
                 'creature_spawn_loc': creature_spawn_loc,
                 'motors_min_spikes': 0, 'motors_max_spikes': motor_max_spikes,
                 'smooth_track_control': True, 'use_sensor': False,
                 'use_dopamine': not ag.no_STDP, 'logging': logging}

# Choose architecture
if ag.architecture == "prebuilt":
    specimen_architecture = architecture.generate_architecture_prebuilt()
elif ag.architecture == "braitenberg":
    specimen_architecture = architecture.generate_architecture_braitenberg()
elif ag.architecture == "all2all":
    specimen_architecture = architecture.generate_architecture_all2all(20, w_min=0, w_max=0.1,
                                d_min=0, d_max=16, hi_pas_p=0.2)
else:
    raise Exception("Wrong network architecture name!!")

# specimen_architecture = architecture.generate_architecture_modified_braitenberg()

x = np.random.randint(int(0.25*WIDTH), int(0.75*WIDTH))
y = np.random.randint(int(0.25*HEIGHT), int(0.75*HEIGHT))
creature_args['creature_spawn_loc'] = [x, y]
environment.spawn_creature(**creature_args)
# Set different cell type if not using plasticity
if ag.no_STDP:
    environment.creatures[0].cell = Cells.GenericCell

environment.creatures[0].create_from_architecture(specimen_architecture, 0, 0)  # No dopamine
environment.creatures[0].add_stim_gate_cell(gate_cells[0], [sensor_cells[0]])
environment.creatures[0].add_stim_gate_cell(gate_cells[1], [sensor_cells[1]])
# Add gaussian noise to gate cells
environment.creatures[0].add_noise_to_cells(gate_cells, mean=noise_mean,
                                             stdev=noise_stdev, gate_cell=True)




time_counter = 0
while True:
    log("stepping")
    environment.step_sim()
    creature = environment.creatures[0]
    # Now check distance to ligth source
    distanceL = creature.body.light_distL
    distanceR = creature.body.light_distR

    log("Sum of distances to targets from sensors:", distanceL, distanceR)
    Lamp, Ramp = map_input(distanceL, distanceR)
    Lgate, Rgate = gate_cells
    creature.stimulate_gate_cell(Lgate, Lamp, creature.neuron2creature_dt)
    creature.stimulate_gate_cell(Rgate, Ramp, creature.neuron2creature_dt)

    log("Creature score=%d"%creature.body.score)


    if target_respawn_time is not None:
        if time_counter % target_respawn_time == 0:
            # Remove and respawn one target
            environment.remove_light_source(np.random.randint(0, len(environment.light_sources)))
            environment.spawn_light_sources(1)

    # Maintain constant amount fo light sources (food)
    log("spawning %d light sources"%(num_lsources - len(environment.light_sources)))
    environment.spawn_light_sources(num_lsources - len(environment.light_sources))


    log("@@@@ CREATURE TIME: %d @@@@"%time_counter)


    if score_after_time is not None and score_after_time == time_counter:
        score = environment.creatures[0].body.score
        log("\n\nScore after %d timesteps: %d\n"%(time_counter, score))
        # Always print score as a bare number so we can get it easily from script
        print score
        break

    time_counter += 1

